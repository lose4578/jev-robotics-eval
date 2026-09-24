'use strict';
const byId = id => document.getElementById(id);
const statusLabels = {success: '成功', failed: '任务失败', error: '执行异常', running: '运行中', pending: '待运行 / 待同步', unknown: '结果未知'};
const filters = ['search', 'environment', 'control-mode', 'experiment', 'information', 'level', 'condition', 'plan-only', 'recovery', 'status', 'mode', 'selection', 'replay'];
const controlLabels = {hierarchical_staged: '分层控制 · JEV 阶段 → 动作', atomic: '原子动作 · 意图 / 视觉方向', direct: '直接控制 · 基础动作', oracle_waypoints: 'Oracle 路标辅助', unknown: '控制模式未标注'};
let snapshot = null;
let previousPayload = '';
let visibleLimit = 60;
let fetching = false;

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined && content !== null) node.textContent = String(content);
  return node;
}
function dateText(value) {
  if (!value) return '暂无时间';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? '暂无时间' : date.toLocaleString('zh-CN', {hour12: false});
}
function levelText(run) {
  const labels = {0: '无特权', 1: '位置真值', 2: '交互 / 进度', 3: '阶段 / 路标'};
  if (run.condition === 'L3-P') return 'L3-P · 仅计划/路标（oracle 辅助）';
  return run.privilege_level === null ? '等级未标注' : `L${run.privilege_level} · ${labels[run.privilege_level]}`;
}
function recoveryText(run) {
  const recovery = run.stuck_recovery ?? run.config?.stuck_recovery ?? 'none';
  return `脱困 ${recovery} · 干预 ${run.intervention_count ?? 0} 次`;
}
function environmentText(run) {
  const environment = run.environment ?? run.config?.environment ?? 'metaworld';
  return {metaworld: 'MetaWorld', robotwin: 'RoboTwin'}[environment] || environment;
}
function samplingText(run) {
  const selection = run.action_selection ?? run.config?.action_selection ?? 'unknown';
  const temperature = run.sampling_temperature ?? run.config?.sampling_temperature;
  return `${selection}${temperature !== null && temperature !== undefined ? ` · T=${temperature}` : ''}`;
}
function controlMode(run) { return run.control_mode ?? run.config?.control_mode ?? 'unknown'; }
function controlText(run) { return controlLabels[controlMode(run)] || controlLabels.unknown; }
function matchesControl(run, selected) {
  return selected === 'all' || selected === controlMode(run) ||
    (selected === 'hierarchical' && ['hierarchical_staged', 'atomic'].includes(controlMode(run)));
}
function mediaUrl(value) {
  if (typeof value !== 'string' || !value.startsWith('/runs/gifs/')) return null;
  const url = new URL(value, window.location.origin);
  return url.origin === window.location.origin && /\.(gif|png)$/i.test(url.pathname) ? url.href : null;
}
function badge(text, kind) { return element('span', `badge ${kind}`, text); }
function setOptions(id, values) {
  const select = byId(id);
  const previous = select.value;
  select.replaceChildren(element('option', '', '全部实验'));
  select.firstChild.value = 'all';
  for (const value of values) {
    const option = element('option', '', value);
    option.value = value;
    select.append(option);
  }
  select.value = values.includes(previous) ? previous : 'all';
}
function matches(run) {
  const terms = byId('search').value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = `${environmentText(run)} ${run.experiment} ${run.task} ${run.mode} seed${run.seed} episode${run.episode_index} policy_seed${run.policy_seed} ${samplingText(run)} ${levelText(run)} ${controlText(run)} ${recoveryText(run)}`.toLowerCase();
  return terms.every(term => haystack.includes(term)) &&
    (byId('environment').value === 'all' || byId('environment').value === (run.environment ?? 'metaworld')) &&
    matchesControl(run, byId('control-mode').value) &&
    (byId('experiment').value === 'all' || byId('experiment').value === run.experiment) &&
    (byId('information').value === 'all' || byId('information').value === run.information) &&
    (byId('level').value === 'all' || byId('level').value === String(run.privilege_level ?? 'unknown')) &&
    (byId('condition').value === 'all' || byId('condition').value === (run.condition ?? (run.privilege_level === null ? 'unknown' : `L${run.privilege_level}`))) &&
    (byId('plan-only').value === 'all' || byId('plan-only').value === String(run.plan_only ?? false)) &&
    (byId('recovery').value === 'all' || byId('recovery').value === (run.stuck_recovery ?? 'none')) &&
    (byId('status').value === 'all' || byId('status').value === run.status) &&
    (byId('mode').value === 'all' || byId('mode').value === run.mode) &&
    (byId('selection').value === 'all' || byId('selection').value === run.action_selection) &&
    (byId('replay').value === 'all' || (byId('replay').value === 'yes') === Boolean(run.gif_url));
}
function configText(config) {
  const labels = {condition: '条件', plan_only: '仅计划/路标', stuck_recovery: '脱困辅助', recovery_window: '检测窗口', recovery_displacement_m: '位移阈值(m)', recovery_cooldown: '冷却', recovery_max_interventions: '最大干预', recovery_steps: '干预步数', mode: '模式', sensor_policy: '策略', action_selection: '动作选择', sampling_temperature: '采样温度', guidance: '引导', action_repeat: '重复', move_scale: '步幅', max_decisions: '决策上限', task_index: '任务索引', image_size: '图像'};
  return Object.entries(labels).filter(([key]) => config[key] !== undefined)
    .map(([key, label]) => `${label} ${config[key]}`).join(' · ');
}
function overview() {
  const counts = snapshot.counts;
  const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
  const items = [
    ['实验记录', total, `${snapshot.experiments.length} 个实验 · 分组比较`, ''],
    ['成功', counts.success || 0, '有效完成的成功回合', 'success'],
    ['任务失败', counts.failed || 0, '有效完成但未成功', 'failed'],
    ['运行中', counts.running || 0, `另有 ${counts.pending || 0} 个待运行 / 待同步`, 'running'],
    ['执行异常', counts.error || 0, '含零动作 · 不计成功率', 'error']
  ];
  byId('overview').replaceChildren(...items.map(([label, value, note, kind]) => {
    const card = element('div', `metric ${kind}`);
    card.append(element('span', 'metric-label', label), element('span', 'metric-value', value), element('span', 'metric-note', note));
    return card;
  }));
}
function renderGroups(matched) {
  const keys = new Set(matched.map(run => `${run.experiment}\u0000${run.privilege_level}\u0000${run.configuration}`));
  const groups = snapshot.groups.filter(group => keys.has(`${group.experiment}\u0000${group.privilege_level}\u0000${group.configuration}`));
  byId('group-count').textContent = `${groups.length} 个实验配置 / 等级组合`;
  const rows = groups.map(group => {
    const row = element('tr');
    const identity = element('td');
    identity.append(element('span', 'experiment-name', `${environmentText(group)} · ${group.experiment}`), element('span', 'config-line', `${configText(group.config)} · #${group.configuration}`));
    const repeated = (group.task_seeds || []).filter(item => item.total > 1);
    if (repeated.length) {
      const details = element('details', 'repeat-details');
      details.append(element('summary', '', `${repeated.length} 组任务 / env seed 的重复回合`));
      for (const item of repeated) {
        const states = [['error', '异常'], ['running', '运行中'], ['pending', '待运行'], ['unknown', '未知']]
          .filter(([key]) => item[key]).map(([key, label]) => `${label} ${item[key]}`).join(' · ');
        details.append(element('p', '', `${item.task} · env seed ${item.seed}: ${item.success}/${item.denominator} 成功 · ${item.denominator}/${item.total} 有效完成 · 干预 ${item.intervention_count ?? 0} 次${states ? ` · ${states}` : ''}`));
      }
      identity.append(details);
    }
    const level = element('td');
    level.append(badge(levelText(group), `level level-${group.privilege_level}`));
    const control = element('td');
    control.append(badge(controlText(group), 'control'));
    const rate = element('td');
    rate.append(element('span', 'rate', group.success_rate === null ? '—' : `${(group.success_rate * 100).toFixed(1)}%`), element('span', 'denominator', `${group.success} / ${group.denominator}`));
    const completed = element('td', '', `${group.denominator} / ${group.total}`);
    const other = element('td', 'mini-status');
    other.textContent = [['running', '运行中'], ['pending', '待运行 / 同步'], ['error', '异常'], ['unknown', '未知'], ['media_only', '仅回放']].filter(([key]) => group[key]).map(([key, label]) => `${label} ${group[key]}`).join(' · ') || '全部完成';
    if (group.intervention_count) other.append(element('div', '', `脱困干预 ${group.intervention_count} 次`));
    row.append(identity, control, level, rate, completed, other, element('td', 'muted', dateText(group.updated_at)));
    return row;
  });
  byId('groups').replaceChildren(...rows);
}
function openPlayer(run) {
  const url = mediaUrl(run.gif_url);
  if (!url) return;
  byId('player-title').textContent = `${environmentText(run)} · ${run.task} · env seed ${run.seed ?? '未知'} · 第 ${run.episode_index ?? 1} 回合`;
  byId('player-caption').textContent = `${run.experiment} · ${controlText(run)} · ${levelText(run)} · ${samplingText(run)} · ${recoveryText(run)} · policy seed ${run.policy_seed ?? '未标注'} · ${statusLabels[run.status]}${run.media_only ? ' · 仅历史回放，未计入成功率' : ''}`;
  byId('player-image').src = url;
  byId('player-download').href = url;
  byId('player').showModal();
}
function renderCard(run) {
  const card = element('article', 'card');
  const gif = mediaUrl(run.gif_url);
  const preview = gif ? element('button', 'preview') : element('div', 'preview');
  if (gif) {
    preview.type = 'button';
    preview.setAttribute('aria-label', `播放 ${environmentText(run)} ${run.task} env seed ${run.seed} 第 ${run.episode_index ?? 1} 回合的 GIF 回放`);
    preview.addEventListener('click', () => openPlayer(run));
    const img = element('img');
    img.src = mediaUrl(run.preview_url) || gif;
    img.alt = `${environmentText(run)} ${run.task} 回放预览`;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.addEventListener('error', () => {
      img.hidden = true;
      preview.append(element('span', 'placeholder', '预览暂不可用，点击打开 GIF'));
    }, {once: true});
    preview.append(img, element('span', 'play', '▶'));
  } else {
    const placeholder = element('div', 'placeholder');
    placeholder.append(element('span', 'placeholder-symbol', run.status === 'running' ? '◌' : '◇'), element('span', '', run.status === 'running' ? '任务运行中，等待回放' : '暂无 GIF 回放'));
    preview.append(placeholder);
  }
  const content = element('div', 'card-content');
  const top = element('div', 'card-top');
  top.append(badge(run.issue === 'zero_actions' ? '零动作异常' : statusLabels[run.status], run.status), badge(levelText(run), `level level-${run.privilege_level}`));
  top.append(badge(controlText(run), 'control'));
  const metadata = element('div', 'metadata');
  metadata.append(element('span', '', `env seed ${run.seed ?? '?'}`), element('span', '', `第 ${run.episode_index ?? 1} 回合`), element('span', '', run.mode));
  metadata.append(element('span', '', samplingText(run)), element('span', '', recoveryText(run)));
  if (run.policy_seed !== null && run.policy_seed !== undefined) metadata.append(element('span', '', `policy seed ${run.policy_seed}`));
  if (run.stuck_recovery === 'jitter' && run.recovery_seed !== null && run.recovery_seed !== undefined) metadata.append(element('span', '', `recovery seed ${run.recovery_seed}`));
  if (run.decisions !== null) metadata.append(element('span', '', `${run.decisions} 次决策`));
  if (run.simulator_steps !== null) metadata.append(element('span', '', `${run.simulator_steps} 步`));
  if (run.wall_seconds !== null) metadata.append(element('span', '', `${run.wall_seconds}s`));
  content.append(top, element('h3', '', `${environmentText(run)} · ${run.task}`), element('p', 'card-experiment', run.experiment), metadata);
  const notes = [];
  if (run.media_only) notes.push('仅历史回放，未计入成功率');
  if (run.privilege_source === 'legacy') notes.push('等级由旧配置推断');
  if (run.status === 'error') notes.push('执行异常，未计入任务失败');
  if (run.artifact_error) notes.push('回放导出异常');
  if (notes.length) content.append(element('p', 'card-note', notes.join(' · ')));
  content.append(element('p', 'card-time', `更新于 ${dateText(run.updated_at)}`));
  card.append(preview, content);
  return card;
}
function render() {
  if (!snapshot) return;
  const matched = snapshot.runs.filter(matches);
  byId('run-count').textContent = `${matched.length} 条匹配记录 · ${matched.filter(run => run.gif_url).length} 个回放`;
  renderGroups(matched);
  byId('runs').replaceChildren(...matched.slice(0, visibleLimit).map(renderCard));
  byId('empty').hidden = matched.length !== 0;
  byId('show-more').hidden = matched.length <= visibleLimit;
}
function updateWarning(connectionError = false) {
  const warnings = [];
  if (connectionError) warnings.push('连接中断，保留上次成功读取的结果；稍后自动重试。');
  if (snapshot?.stale_file_count) warnings.push(`${snapshot.stale_file_count} 个结果文件正在写入或暂不可读，已保留它们的上次有效快照。`);
  if (snapshot?.stale_snapshot) warnings.push('结果目录暂时变化，当前展示上次有效快照。');
  byId('warning').textContent = warnings.join(' ');
  byId('warning').hidden = warnings.length === 0;
}
async function refresh() {
  if (fetching) return;
  fetching = true;
  byId('refresh').disabled = true;
  try {
    const response = await fetch('/api/runs', {cache: 'no-store', signal: AbortSignal.timeout(12000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data.runs) || !Array.isArray(data.groups)) throw new Error('Invalid response');
    snapshot = data;
    const stable = JSON.stringify([data.runs, data.groups, data.counts, data.experiments]);
    if (stable !== previousPayload) {
      previousPayload = stable;
      setOptions('experiment', data.experiments);
      overview();
      render();
    }
    byId('last-updated').textContent = `结果最后更新：${dateText(data.updated_at)}`;
    byId('last-polled').textContent = `最近同步：${dateText(data.generated_at)} · 5 秒自动刷新`;
    byId('connection').textContent = '实时同步';
    byId('connection-dot').className = 'dot online';
    updateWarning();
  } catch (error) {
    byId('connection').textContent = '连接中断';
    byId('connection-dot').className = 'dot offline';
    updateWarning(true);
  } finally {
    fetching = false;
    byId('refresh').disabled = false;
  }
}
const requestedControl = new URL(window.location.href).searchParams.get('control-mode');
if ([...byId('control-mode').options].some(option => option.value === requestedControl)) byId('control-mode').value = requestedControl;
for (const id of filters) byId(id).addEventListener(id === 'search' ? 'input' : 'change', () => {
  if (id === 'control-mode') {
    const url = new URL(window.location.href);
    if (byId(id).value === 'all') url.searchParams.delete(id);
    else url.searchParams.set(id, byId(id).value);
    window.history.replaceState(null, '', url);
  }
  visibleLimit = 60;
  render();
});
byId('refresh').addEventListener('click', refresh);
byId('show-more').addEventListener('click', () => {visibleLimit += 60; render();});
byId('close-player').addEventListener('click', () => byId('player').close());
byId('player').addEventListener('close', () => byId('player-image').removeAttribute('src'));
byId('player').addEventListener('click', event => {if (event.target === byId('player')) byId('player').close();});
refresh();
setInterval(refresh, 5000);
