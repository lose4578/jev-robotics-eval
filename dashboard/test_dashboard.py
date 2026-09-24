"""Isolated checks for aggregation, concurrent writes, and HTTP file boundaries."""
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("serve_dashboard", PROJECT / "scripts" / "serve_dashboard.py")
DASHBOARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DASHBOARD)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "runs" / "gifs").mkdir(parents=True)
        self.snapshot = DASHBOARD.Snapshot(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, value):
        path = self.root / "runs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def summary(self, name="experiment", config=None, results=None):
        return self.write(name + "/summary.json", {
            "config": config or {"mode": "vision", "privilege_level": 0},
            "results": results if results is not None else [
                {"task": "reach-v3", "seed": 2, "status": "completed", "success": True, "decisions": 5}
            ],
        })

    def test_privilege_explicit_and_legacy(self):
        cases = [({"privilege_level": 1}, 1, "explicit"),
                 ({"privilege_level": "L0", "information": "privileged"}, 0, "explicit"),
                 ({"information": "nonprivileged", "guidance": "waypoints"}, 0, "legacy"),
                 ({"guidance": "waypoints"}, 3, "legacy"),
                 ({"information": "privileged"}, 2, "legacy"),
                 ({}, 2, "legacy")]
        for config, level, source in cases:
            self.assertEqual(DASHBOARD.privilege(config), (level, source))
        self.assertEqual(DASHBOARD.privilege({}, legacy_default=False), (None, "unknown"))

    def test_control_mode_follows_controller_precedence(self):
        cases = [
            ({"action_space": "metaworld_atomic", "sensor_policy": "staged"}, 1, "atomic"),
            ({"action_space": "atomic", "guidance": "waypoints"}, 3, "atomic"),
            ({"sensor_policy": "staged", "guidance": "direct"}, 0, "hierarchical_staged"),
            ({"sensor_policy": "staged", "guidance": "direct"}, 2, "hierarchical_staged"),
            ({"sensor_policy": "staged", "guidance": "waypoints"}, 3, "oracle_waypoints"),
            ({"sensor_policy": "staged", "plan_only": True}, 3, "oracle_waypoints"),
            ({"sensor_policy": "staged"}, 3, "oracle_waypoints"),
            ({"sensor_policy": "staged", "guidance": "waypoints"}, 0, "hierarchical_staged"),
            ({"sensor_policy": "direct"}, 0, "direct"),
            ({"guidance": "direct"}, 2, "direct"),
            ({"action_space": "future", "sensor_policy": "staged"}, 1, "unknown"),
            ({"control_mode": "hierarchical_staged"}, None, "unknown"),
            ({}, None, "unknown"),
        ]
        for config, level, expected in cases:
            with self.subTest(config=config, level=level):
                self.assertEqual(DASHBOARD.control_mode(config, level), expected)

    def test_control_modes_separate_groups_and_preserve_denominators(self):
        self.summary(config={"privilege_level": 1, "sensor_policy": "staged"}, results=[
            {"task": "a", "seed": 2, "success": True, "action_space": "metaworld_atomic"},
            {"task": "b", "seed": 2, "success": False, "action_space": "primitive"},
            {"task": "c", "seed": 2, "status": "error", "action_space": "primitive"},
        ])
        data = self.snapshot.get()
        groups = {group["control_mode"]: group for group in data["groups"]}
        self.assertEqual(set(groups), {"atomic", "hierarchical_staged"})
        self.assertEqual((groups["atomic"]["success"], groups["atomic"]["denominator"]), (1, 1))
        self.assertEqual((groups["hierarchical_staged"]["success"], groups["hierarchical_staged"]["denominator"]), (0, 1))
        for item in data["runs"] + data["groups"]:
            self.assertEqual(item["control_mode"], item["config"]["control_mode"])
        for group in data["groups"]:
            self.assertTrue(all(item["control_mode"] == group["control_mode"] for item in group["task_seeds"]))

    def test_errors_and_zero_actions_do_not_enter_denominator(self):
        results = [
            {"task": "a", "status": "completed", "success": True, "decisions": 3},
            {"task": "b", "status": "completed", "success": False, "decisions": 7},
            {"task": "c", "status": "error", "success": False, "decisions": 0},
            {"task": "d", "status": "completed", "success": False, "simulator_steps": 0},
        ]
        self.summary(results=results)
        data = self.snapshot.get()
        self.assertEqual(data["counts"], {"success": 1, "failed": 1, "error": 2})
        group = data["groups"][0]
        self.assertEqual((group["success"], group["denominator"], group["total"]), (1, 2, 4))
        self.assertEqual(group["success_rate"], 0.5)

    def test_experiments_and_configurations_have_separate_rates(self):
        self.summary("one", results=[
            {"task": "a", "success": True, "mode": "vision"},
            {"task": "b", "success": False, "mode": "text"},
        ])
        self.summary("two")
        data = self.snapshot.get()
        self.assertEqual(len(data["groups"]), 3)
        self.assertTrue(all(group["denominator"] == 1 for group in data["groups"]))

    def test_legacy_environment_defaults_to_metaworld(self):
        self.summary(results=[
            {"task": "shared-task", "seed": 2, "episode_index": 1, "success": True, "decisions": 3},
            {"environment": "metaworld", "task": "shared-task", "seed": 2, "episode_index": 2,
             "success": False, "decisions": 3},
        ])
        data = self.snapshot.get()
        self.assertEqual({run["environment"] for run in data["runs"]}, {"metaworld"})
        self.assertTrue(all(run["config"]["environment"] == "metaworld" for run in data["runs"]))
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["denominator"], 2)

    def test_legacy_l3_defaults_to_full_information_without_recovery(self):
        self.summary(config={"privilege_level": 3}, results=[
            {"task": "shared-task", "seed": 2, "episode_index": 1, "success": True, "decisions": 3},
            {"task": "shared-task", "seed": 2, "episode_index": 2, "plan_only": False, "condition": "L3",
             "stuck_recovery": "none", "intervention_count": 0,
             "success": False, "decisions": 3},
            {"task": "shared-task", "seed": 2, "episode_index": 3, "condition": "L3-P",
             "success": False, "decisions": 3},
        ])
        data = self.snapshot.get()
        self.assertEqual({run["condition"] for run in data["runs"]}, {"L3"})
        self.assertEqual({run["plan_only"] for run in data["runs"]}, {False})
        self.assertEqual({run["stuck_recovery"] for run in data["runs"]}, {"none"})
        self.assertEqual({run["intervention_count"] for run in data["runs"]}, {0})
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["denominator"], 3)

    def test_l0_l3p_and_full_l3_keep_separate_five_episode_rates(self):
        results = []
        for condition, level, plan_only, successes in [("L0", 0, False, 0), ("L3-P", 3, True, 3),
                                                       ("L3", 3, False, 5)]:
            for episode in range(1, 6):
                results.append({"task": "shared-task", "seed": 2, "episode_index": episode,
                                "policy_seed": 99 + episode, "privilege_level": level, "plan_only": plan_only,
                                "condition": condition, "status": "completed", "success": episode <= successes,
                                "decisions": 10, "action_selection": "sample", "sampling_temperature": 1.0})
        self.summary(config={"mode": "vision", "episodes": 5, "privilege_level": 0}, results=results)
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 15)
        groups = {group["condition"]: group for group in data["groups"]}
        self.assertEqual(set(groups), {"L0", "L3-P", "L3"})
        self.assertEqual(len({group["configuration"] for group in groups.values()}), 3)
        for condition, successes in [("L0", 0), ("L3-P", 3), ("L3", 5)]:
            group = groups[condition]
            self.assertEqual((group["success"], group["denominator"], group["total"]), (successes, 5, 5))
            self.assertEqual(len(group["task_seeds"]), 1)
            self.assertEqual(group["task_seeds"][0]["condition"], condition)
            self.assertEqual(group["task_seeds"][0]["denominator"], 5)
        self.assertEqual(groups["L3-P"]["privilege_level"], 3)
        self.assertEqual(groups["L3-P"]["label"], "L3-P · 仅计划/路标（oracle 辅助）")
        self.assertTrue(all(run["information"] == "privileged" for run in data["runs"]
                            if run["condition"] == "L3-P"))

    def test_l3p_matrix_config_applies_to_completed_and_running_episodes(self):
        child = self.summary("batch/job1", config={"mode": "vision", "privilege_level": 3})
        self.write("batch/matrix.json", {"config": {"mode": "vision", "privilege_level": 3,
                                                     "plan_only": True, "condition": "L3-P", "episodes": 5}, "jobs": [
            {"task": "reach-v3", "seed": 2, "summary": str(child), "status": "completed", "episode_index": 1},
            {"task": "reach-v3", "seed": 2, "summary": str(child.parent.parent / "job2" / "summary.json"),
             "status": "running", "episode_index": 2},
        ]})
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 2)
        self.assertEqual({run["condition"] for run in data["runs"]}, {"L3-P"})
        self.assertEqual({run["plan_only"] for run in data["runs"]}, {True})
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual((data["groups"][0]["denominator"], data["groups"][0]["running"]), (1, 1))

    def test_recovery_modes_are_separate_but_intervention_counts_do_not_split_groups(self):
        results = []
        for recovery in ["none", "jitter"]:
            for episode in range(1, 6):
                results.append({"task": "shared-task", "seed": 2, "episode_index": episode,
                                "policy_seed": 99 + episode, "stuck_recovery": recovery,
                                "intervention_count": episode - 1 if recovery == "jitter" else 0,
                                "status": "completed", "success": recovery == "jitter", "decisions": 10})
        self.summary(config={"mode": "vision", "privilege_level": 3, "plan_only": True}, results=results)
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 10)
        groups = {group["stuck_recovery"]: group for group in data["groups"]}
        self.assertEqual(set(groups), {"none", "jitter"})
        self.assertEqual((groups["none"]["success"], groups["none"]["denominator"]), (0, 5))
        self.assertEqual((groups["jitter"]["success"], groups["jitter"]["denominator"]), (5, 5))
        self.assertEqual(groups["none"]["intervention_count"], 0)
        self.assertEqual(groups["jitter"]["intervention_count"], 10)
        self.assertEqual(groups["jitter"]["task_seeds"][0]["intervention_count"], 10)
        self.assertEqual({run["intervention_count"] for run in data["runs"] if run["stuck_recovery"] == "jitter"},
                         {0, 1, 2, 3, 4})

    def test_jitter_defaults_and_parameter_overrides_define_groups_but_seeds_do_not(self):
        variants = [{}, dict(DASHBOARD.RECOVERY_DEFAULTS), {"recovery_window": 8},
                    {"recovery_displacement_m": 0.004}, {"recovery_cooldown": 8},
                    {"recovery_max_interventions": 5}, {"recovery_steps": 4}]
        jobs = []
        for episode, parameters in enumerate(variants, start=1):
            child = self.summary(f"batch/job{episode}", config={
                "mode": "vision", "privilege_level": 3, "plan_only": True, "stuck_recovery": "jitter",
                "episode_index": episode, "recovery_seed": 200 + episode, **parameters,
            }, results=[{"task": "shared-task", "seed": 2, "episode_index": episode, "recovery_seed": 200 + episode,
                         "status": "completed", "success": True, "decisions": 10}])
            jobs.append({"task": "shared-task", "seed": 2, "episode_index": episode,
                         "summary": str(child), "status": "completed"})
        self.write("batch/matrix.json", {"config": {"mode": "vision", "privilege_level": 3}, "jobs": jobs})
        data = self.snapshot.get()
        self.assertEqual(len(data["groups"]), 6)
        self.assertEqual(sorted(group["denominator"] for group in data["groups"]), [1, 1, 1, 1, 1, 2])
        self.assertEqual(len({run["recovery_seed"] for run in data["runs"]}), 7)
        default_group = next(group for group in data["groups"] if group["denominator"] == 2)
        for key, default in DASHBOARD.RECOVERY_DEFAULTS.items():
            self.assertEqual(default_group["config"][key], default)

    def test_no_recovery_ignores_tuning_parameters_and_effective_seed(self):
        jobs = []
        for episode in range(1, 6):
            child = self.summary(f"batch/job{episode}", config={
                "privilege_level": 3, "plan_only": True, "stuck_recovery": "none",
                "episode_index": episode, "recovery_seed": 200 + episode,
                "recovery_window": 6 + episode, "recovery_displacement_m": 0.003 * episode,
                "recovery_cooldown": 6 + episode, "recovery_max_interventions": 3 + episode,
                "recovery_steps": 2 + episode,
            }, results=[{"task": "shared-task", "seed": 2, "episode_index": episode,
                         "status": "completed", "success": True, "decisions": 10}])
            jobs.append({"task": "shared-task", "seed": 2, "episode_index": episode,
                         "summary": str(child), "status": "completed"})
        self.write("batch/matrix.json", {"config": {"privilege_level": 3}, "jobs": jobs})
        data = self.snapshot.get()
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual(data["groups"][0]["denominator"], 5)
        self.assertEqual({run["recovery_seed"] for run in data["runs"]}, {201, 202, 203, 204, 205})
        self.assertFalse(set(DASHBOARD.RECOVERY_DEFAULTS) & data["groups"][0]["config"].keys())

    def test_same_task_seed_and_episode_in_different_environments_have_separate_rates(self):
        self.summary(config={"environment": "metaworld", "privilege_level": 0, "mode": "vision",
                             "action_selection": "sample", "sampling_temperature": 1.0}, results=[
            {"environment": environment, "task": "shared-task", "seed": 2, "episode_index": 1,
             "policy_seed": 100, "status": "completed", "success": success, "decisions": 10}
            for environment, success in [("metaworld", True), ("robotwin", False)]
        ])
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 2)
        self.assertEqual(len({run["configuration"] for run in data["runs"]}), 2)
        groups = {group["environment"]: group for group in data["groups"]}
        self.assertEqual(set(groups), {"metaworld", "robotwin"})
        self.assertEqual((groups["metaworld"]["success"], groups["metaworld"]["denominator"]), (1, 1))
        self.assertEqual((groups["robotwin"]["success"], groups["robotwin"]["denominator"]), (0, 1))
        for environment, group in groups.items():
            self.assertEqual(group["task_seeds"][0]["total"], 1)
            self.assertEqual(group["task_seeds"][0]["environment"], environment)

    def test_robotwin_environment_is_inherited_from_config_and_matrix(self):
        self.summary("serial", config={"environment": "robotwin", "privilege_level": 0})
        child = self.summary("batch/job1")
        self.write("batch/matrix.json", {"config": {"environment": "robotwin", "privilege_level": 0}, "jobs": [
            {"task": "reach-v3", "seed": 2, "summary": str(child), "status": "completed", "success": True}
        ]})
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 2)
        self.assertEqual({run["environment"] for run in data["runs"]}, {"robotwin"})
        self.assertEqual({group["environment"] for group in data["groups"]}, {"robotwin"})

    def test_five_episodes_with_the_same_task_seed_are_counted_individually(self):
        self.summary(config={"tasks": ["reach-v3"], "seeds": [2], "episodes": 5,
                             "privilege_level": 0, "action_selection": "sample", "sampling_temperature": 1.0,
                             "policy_seed": 100}, results=[
            {"task": "reach-v3", "seed": 2, "episode_index": episode, "policy_seed": 99 + episode,
             "action_selection": "sample", "sampling_temperature": 1.0,
             "status": "completed", "success": episode <= 3, "decisions": 20}
            for episode in range(1, 6)
        ])
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 5)
        self.assertEqual(len({run["id"] for run in data["runs"]}), 5)
        self.assertEqual({run["episode_index"] for run in data["runs"]}, {1, 2, 3, 4, 5})
        self.assertEqual({run["policy_seed"] for run in data["runs"]}, {100, 101, 102, 103, 104})
        self.assertEqual(len(data["groups"]), 1)
        group = data["groups"][0]
        self.assertEqual((group["success"], group["failed"], group["denominator"], group["total"]), (3, 2, 5, 5))
        self.assertEqual(group["success_rate"], 0.6)
        pair = group["task_seeds"][0]
        self.assertEqual((pair["task"], pair["seed"], pair["success"], pair["denominator"], pair["total"]),
                         ("reach-v3", 2, 3, 5, 5))

    def test_partial_five_episode_summary_adds_only_missing_episodes(self):
        self.summary(config={"tasks": ["reach-v3"], "seeds": [2], "episodes": 5,
                             "privilege_level": 0, "action_selection": "sample", "policy_seed": 100},
                     results=[{"task": "reach-v3", "seed": 2, "episode_index": episode,
                               "policy_seed": 99 + episode, "success": True, "decisions": 10}
                              for episode in [1, 2]])
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 5)
        self.assertEqual(data["counts"], {"success": 2, "pending": 3})
        pending = [run for run in data["runs"] if run["status"] == "pending"]
        self.assertEqual([run["episode_index"] for run in pending], [3, 4, 5])
        self.assertEqual([run["policy_seed"] for run in pending], [102, 103, 104])
        group = data["groups"][0]
        self.assertEqual((group["denominator"], group["total"]), (2, 5))

    def test_parallel_five_episode_configs_do_not_split_on_policy_seed(self):
        jobs = []
        for episode in range(1, 6):
            summary = self.summary(f"batch/job{episode}", config={
                "tasks": ["reach-v3"], "seeds": [2], "episodes": 1, "episode_index": episode,
                "policy_seed": 99 + episode, "privilege_level": 0,
                "action_selection": "sample", "sampling_temperature": 0.8,
            }, results=[{"task": "reach-v3", "seed": 2, "episode_index": episode, "policy_seed": 99 + episode,
                         "status": "completed", "success": episode <= 3, "decisions": 10}])
            jobs.append({"task": "reach-v3", "seed": 2, "episode_index": episode, "policy_seed": 99 + episode,
                         "summary": str(summary), "status": "completed", "success": episode <= 3})
        self.write("batch/matrix.json", {"config": {"episodes": 5, "policy_seed": 100, "privilege_level": 0,
                                                     "action_selection": "sample", "sampling_temperature": 0.8},
                                           "jobs": jobs})
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 5)
        self.assertEqual(len(data["groups"]), 1)
        self.assertEqual({run["episodes"] for run in data["runs"]}, {5})
        self.assertEqual({run["policy_seed"] for run in data["runs"]}, {100, 101, 102, 103, 104})
        self.assertEqual((data["groups"][0]["success"], data["groups"][0]["denominator"]), (3, 5))

    def test_strategy_temperature_and_privilege_never_share_a_rate(self):
        variants = [("sample", 1.0, 0), ("sample", 1.0, 0), ("argmax", 1.0, 0),
                    ("sample", 0.5, 0), ("sample", 1.0, 1)]
        self.summary(results=[
            {"task": "reach-v3", "seed": 2, "episode_index": episode, "policy_seed": episode,
             "action_selection": selection, "sampling_temperature": temperature, "privilege_level": level,
             "status": "completed", "success": True, "decisions": 4}
            for episode, (selection, temperature, level) in enumerate(variants, start=1)
        ])
        data = self.snapshot.get()
        self.assertEqual(len(data["groups"]), 4)
        self.assertEqual(sorted(group["denominator"] for group in data["groups"]), [1, 1, 1, 2])

    def test_scheduler_error_does_not_erase_completed_child_episode(self):
        summary = self.summary("batch/job1")
        self.write("batch/matrix.json", {"config": {"privilege_level": 0}, "jobs": [
            {"task": "reach-v3", "seed": 2, "summary": str(summary), "status": "error", "error": "export failed"}
        ]})
        data = self.snapshot.get()
        self.assertEqual(data["counts"], {"success": 1})
        self.assertEqual(data["groups"][0]["denominator"], 1)

    def test_partial_writes_keep_last_valid_json_then_refresh(self):
        path = self.summary()
        original = self.snapshot.get()
        path.write_text('{"config":')
        self.snapshot.checked_at = 0
        partial = self.snapshot.get()
        self.assertEqual(partial["runs"], original["runs"])
        self.assertEqual(partial["stale_file_count"], 1)
        self.summary(results=[{"task": "new-task", "success": False, "decisions": 8}])
        self.snapshot.checked_at = 0
        refreshed = self.snapshot.get()
        self.assertEqual(refreshed["runs"][0]["task"], "new-task")
        self.assertEqual(refreshed["stale_file_count"], 0)

    def test_nested_matrix_and_child_summary_count_each_job_once(self):
        child = self.summary("batch/job1")
        jobs = []
        for number, state in enumerate(["completed", "running", "queued", "error"], start=1):
            jobs.append({"task": "reach-v3", "seed": number, "status": state,
                         "summary": str(self.root / "runs" / "batch" / f"job{number}" / "summary.json"),
                         "success": state == "completed"})
        self.write("batch/matrix.json", {"config": {"privilege_level": 0}, "jobs": jobs})
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 4)
        self.assertEqual(data["experiments"], ["batch"])
        self.assertEqual(data["counts"], {"success": 1, "running": 1, "pending": 1, "error": 1})
        self.assertEqual(sum(group["denominator"] for group in data["groups"]), 1)

    def test_media_sidecar_and_summary_link_are_deduplicated(self):
        gif = self.root / "runs/gifs/batch/reach-v3-vision-seed2-success.gif"
        gif.parent.mkdir()
        gif.write_bytes(b"GIF89a")
        self.write("gifs/batch/reach-v3-vision-seed2-success.json", {"privilege_level": 1, "environment": "robotwin"})
        self.summary("batch", config={"mode": "vision"}, results=[
            {"task": "reach-v3", "seed": 2, "success": True, "decisions": 4,
             "gif": "runs/gifs/batch/reach-v3-vision-seed2-success.gif"}
        ])
        data = self.snapshot.get()
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(data["runs"][0]["privilege_level"], 1)
        self.assertEqual(data["runs"][0]["environment"], "robotwin")
        self.assertTrue(data["runs"][0]["eligible"])

    def test_api_does_not_publish_credentials_or_raw_errors(self):
        self.summary(config={"privilege_level": 0, "api_key": "secret-value", "jev_url": "https://user:password@host/"},
                     results=[{"task": "<img src=x onerror=alert(1)>", "status": "error", "error": "API secret-value"}])
        encoded = json.dumps(self.snapshot.get())
        self.assertNotIn("secret-value", encoded)
        self.assertNotIn("password", encoded)
        self.assertNotIn("api_key", encoded)
        self.assertEqual(self.snapshot.get()["runs"][0]["task"], "<img src=x onerror=alert(1)>")

    def test_http_only_serves_allowed_files_and_rejects_symlink_escape(self):
        (self.root / "dashboard").mkdir()
        (self.root / "dashboard/index.html").write_text("safe dashboard")
        (self.root / "secret.gif").write_bytes(b"PRIVATE")
        (self.root / "runs/gifs/escape.gif").symlink_to(self.root / "secret.gif")
        (self.root / "runs/gifs/linked").symlink_to(self.root, target_is_directory=True)
        (self.root / "runs/gifs/valid.gif").write_bytes(b"GIF89a")
        self.summary()
        server = DASHBOARD.DashboardServer(("127.0.0.1", 0), DASHBOARD.DashboardHandler)
        server.snapshot = self.snapshot
        server.dashboard_root = self.root / "dashboard"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            for path in ["/runs/experiment/summary.json", "/scripts/serve_dashboard.py", "/.git/config",
                         "/../secret.gif", "/%2e%2e/secret.gif", "/runs/gifs/%2e%2e/secret.gif",
                         "/runs/gifs/escape.gif", "/runs/gifs/linked/secret.gif", "/runs/gifs/a%5cb.gif",
                         "/runs/gifs/valid.gif%00", "/runs/gifs/"]:
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, 404, path)
                self.assertNotIn(b"PRIVATE", response.read())
            for path, expected in [("/", b"safe dashboard"), ("/runs/gifs/valid.gif", b"GIF89a")]:
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), expected)
                self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
