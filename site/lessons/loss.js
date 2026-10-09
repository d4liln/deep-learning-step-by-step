/**
 * Lesson: loss functions (MSE, MAE, binary and categorical cross-entropy) and matching output activations.
 */
(function () {
  /**
   * Creates an SVG text element with a pixel font size, anchor and optional mono / bold / centered options.
   */
  function txt(h, parent, x, y, str, o) {
    o = o || {};
    return h.el('text', {
      x, y, text: str,
      style: 'font-size:' + (o.size || 13) + 'px' + (o.bold ? ';font-weight:700' : ''),
      fill: o.fill || 'var(--ink)',
      'text-anchor': o.anchor || 'middle',
      class: o.mono ? 'mono' : null,
      'dominant-baseline': o.mid ? 'central' : null,
      opacity: o.opacity
    }, parent);
  }

  /**
   * Converts a pointer event into SVG viewBox coordinates.
   */
  function svgPoint(svg, e) {
    const p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  }

  /**
   * Formats an axis tick with a French decimal comma and a true minus sign.
   */
  function tick(v) {
    return String(v).replace('.', ',').replace('-', '−');
  }

  /**
   * Draws a light grid, ticks, tick labels and axis titles inside the pixel box b.
   */
  function axes(h, g, b, sx, sy, o) {
    (o.xt || []).forEach((v) => {
      const x = sx(v);
      if (o.grid) h.el('line', { x1: x, y1: b.y0, x2: x, y2: b.y1, stroke: 'var(--line)' }, g);
      h.el('line', { x1: x, y1: b.y1, x2: x, y2: b.y1 + 5, stroke: 'var(--ink)' }, g);
      txt(h, g, x, b.y1 + 19, tick(v), { size: 12, fill: 'var(--muted)' });
    });
    (o.yt || []).forEach((v) => {
      const y = sy(v);
      if (o.grid) h.el('line', { x1: b.x0, y1: y, x2: b.x1, y2: y, stroke: 'var(--line)' }, g);
      h.el('line', { x1: b.x0 - 5, y1: y, x2: b.x0, y2: y, stroke: 'var(--ink)' }, g);
      txt(h, g, b.x0 - 9, y, tick(v), { size: 12, fill: 'var(--muted)', anchor: 'end', mid: true });
    });
    h.el('line', { x1: b.x0, y1: b.y1, x2: b.x1, y2: b.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    h.el('line', { x1: b.x0, y1: b.y0, x2: b.x0, y2: b.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    if (o.xl) txt(h, g, (b.x0 + b.x1) / 2, b.y1 + 38, o.xl, { size: 13, fill: 'var(--muted)' });
    if (o.yl) txt(h, g, b.x0 - 4, b.y0 - 10, o.yl, { size: 13, fill: 'var(--muted)', anchor: 'start' });
  }

  /**
   * Clicks the i-th button of a segmented toggle (used to script its animation).
   */
  function pick(wrap, i) { wrap.children[i].click(); }

  const BASE = [[1, 1.8], [2, 2.0], [3, 3.1], [4, 3.2], [5, 4.4], [6, 4.5], [7, 5.4], [8, 5.9]];

  /**
   * Mean squared error of the line y = a x + b over the points.
   */
  function mse(pts, a, b) {
    return pts.reduce((s, p) => s + Math.pow(p[1] - (a * p[0] + b), 2), 0) / pts.length;
  }

  /**
   * Mean absolute error of the line y = a x + b over the points.
   */
  function mae(pts, a, b) {
    return pts.reduce((s, p) => s + Math.abs(p[1] - (a * p[0] + b)), 0) / pts.length;
  }

  /**
   * Least-squares line (closed form), minimising the MSE.
   */
  function fitMSE(pts) {
    const n = pts.length;
    const mx = pts.reduce((s, p) => s + p[0], 0) / n;
    const my = pts.reduce((s, p) => s + p[1], 0) / n;
    let num = 0, den = 0;
    pts.forEach((p) => { num += (p[0] - mx) * (p[1] - my); den += (p[0] - mx) * (p[0] - mx); });
    const a = num / den;
    return { a, b: my - a * mx };
  }

  /**
   * Line minimising the MAE, found by a grid search (the MAE has no simple closed form).
   */
  function fitMAE(pts) {
    let best = { a: 0, b: 0, e: Infinity };
    for (let a = -1; a <= 2; a += 0.02) {
      for (let b = -3; b <= 8; b += 0.05) {
        const e = mae(pts, a, b);
        if (e < best.e) best = { a, b, e };
      }
    }
    return best;
  }

  Course.register({
    id: 'loss',
    order: 4,
    title: 'Fonctions de coût',
    summary: 'Comment mesurer l\'erreur d\'un réseau : <b>MSE</b> et <b>MAE</b> pour la régression, <b>entropie croisée</b> binaire et catégorielle pour la classification, et l\'activation de sortie qui va avec.',
    pdfPages: '19',
    steps: [
      {
        title: 'Mesurer l\'erreur',
        duration: 9000,
        text: String.raw`<p>Pour entraîner un réseau, il faut un nombre qui dit <b>à quel point il se trompe</b> : la fonction de coût (<i>loss</i>). L'apprentissage cherche à la minimiser.</p>
<p>Pour la régression (mais aussi, moins souvent, la classification), on compare la prédiction $\hat y$ à la valeur vraie $y$ via l'écart $e = y-\hat y$ :</p>
<ul>
<li><b>MSE</b> (erreur quadratique, perte L2) : $\dfrac1n\sum_i (y_i-\hat y_i)^2$ ;</li>
<li><b>MAE</b> (erreur absolue, perte L1) : $\dfrac1n\sum_i |y_i-\hat y_i|$.</li>
</ul>
<p>Déplacez l'écart $e$ : la parabole grossit beaucoup plus vite que la valeur absolue.</p>`,
        note: String.raw`MSE : les gros écarts sont <b>très</b> pénalisés (écart doublé, pénalité $\times 4$). MAE : la pénalité est proportionnelle à l'écart.`,
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x0: 70, x1: 690, y0: 30, y1: 350 };
          const sx = h.scale(-3, 3, b.x0, b.x1), sy = h.scale(0, 9, b.y1, b.y0);
          axes(h, svg, b, sx, sy, { xt: [-3, -2, -1, 0, 1, 2, 3], yt: [0, 3, 6, 9], grid: true, xl: "écart  e = y − ŷ", yl: 'pénalité' });
          const path = (f) => {
            let d = '';
            for (let i = 0; i <= 120; i++) {
              const e = -3 + 6 * i / 120;
              d += (i ? 'L' : 'M') + sx(e).toFixed(1) + ',' + sy(f(e)).toFixed(1);
            }
            return d;
          };
          h.el('path', { d: path((e) => e * e), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3 }, svg);
          h.el('path', { d: path(Math.abs), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 3 }, svg);
          h.el('line', { x1: b.x0 + 220, y1: b.y0 + 22, x2: b.x0 + 250, y2: b.y0 + 22, stroke: 'var(--accent)', 'stroke-width': 3 }, svg);
          txt(h, svg, b.x0 + 258, b.y0 + 22, 'MSE : e²', { size: 13, anchor: 'start', mid: true });
          h.el('line', { x1: b.x0 + 220, y1: b.y0 + 44, x2: b.x0 + 250, y2: b.y0 + 44, stroke: 'var(--accent-2)', 'stroke-width': 3 }, svg);
          txt(h, svg, b.x0 + 258, b.y0 + 44, 'MAE : |e|', { size: 13, anchor: 'start', mid: true });
          const dyn = h.el('g', {}, svg);
          const out = h.readout('');
          let e = 2;
          const render = () => {
            while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
            h.el('line', { x1: sx(e), y1: b.y0, x2: sx(e), y2: b.y1, stroke: 'var(--ink)', 'stroke-dasharray': '3 4', opacity: 0.6 }, dyn);
            h.el('line', { x1: b.x0, y1: sy(e * e), x2: sx(e), y2: sy(e * e), stroke: 'var(--accent)', 'stroke-dasharray': '3 4' }, dyn);
            h.el('line', { x1: b.x0, y1: sy(Math.abs(e)), x2: sx(e), y2: sy(Math.abs(e)), stroke: 'var(--accent-2)', 'stroke-dasharray': '3 4' }, dyn);
            h.el('circle', { cx: sx(e), cy: sy(e * e), r: 7, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2 }, dyn);
            h.el('circle', { cx: sx(e), cy: sy(Math.abs(e)), r: 7, fill: 'var(--accent-2)', stroke: 'var(--surface)', 'stroke-width': 2 }, dyn);
            out(`$e=${h.fmt(e, 2)}$ &nbsp; $e^2=${h.fmt(e * e, 2)}$ &nbsp; $|e|=${h.fmt(Math.abs(e), 2)}$`);
          };
          const sl = h.slider({ label: 'e', min: -3, max: 3, step: 0.05, value: e, decimals: 2, onInput: (v) => { e = v; render(); } });
          const hit = h.el('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: 'transparent', style: 'cursor:ew-resize' }, svg);
          let drag = false;
          const move = (ev) => {
            e = Math.max(-3, Math.min(3, sx.invert(svgPoint(svg, ev).x)));
            sl.set(e);
            render();
          };
          hit.addEventListener('pointerdown', (ev) => { drag = true; move(ev); });
          const onMove = (ev) => { if (drag) move(ev); };
          const onUp = () => { drag = false; };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
          h.onLeave(() => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); });
          render();
        }
      },
      {
        title: 'Régression : les résidus',
        duration: 12000,
        text: String.raw`<p>Ajustons une droite $\hat y = a\,x + b$ à quelques points. Pour chaque point, le <b>résidu</b> est l'écart vertical $r_i = y_i-\hat y_i$.</p>
<p>La MSE additionne l'<b>aire des carrés</b> construits sur les résidus ; la MAE additionne les <b>longueurs</b> des segments.</p>
<p>Bougez la pente et l'ordonnée à l'origine, ou laissez le calcul trouver la meilleure droite pour chaque coût.</p>`,
        note: String.raw`« Meilleure droite » dépend de la fonction de coût : la droite minimisant la MSE n'est pas exactement celle qui minimise la MAE.`,
        check: {
          q: 'Un résidu passe de 1 à 3 (il est triplé). De combien est multipliée sa contribution à la MSE, et à la MAE ?',
          choices: ['MSE × 3, MAE × 3', 'MSE × 9, MAE × 3', 'MSE × 3, MAE × 9'],
          answer: 1,
          explain: String.raw`MSE : $3^2 = 9$ fois plus. MAE : $|3|/|1| = 3$ fois plus.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x0: 60, x1: 690, y0: 20, y1: 350 };
          const sx = h.scale(0, 10, b.x0, b.x1), sy = h.scale(0, 10, b.y1, b.y0);
          const unit = (b.y1 - b.y0) / 10;
          const clipId = 'loss-clip-reg';
          const cp = h.el('clipPath', { id: clipId }, svg);
          h.el('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0 }, cp);
          axes(h, svg, b, sx, sy, { xt: [0, 2, 4, 6, 8, 10], yt: [0, 2, 4, 6, 8, 10], grid: true, xl: 'x', yl: 'y' });
          const dyn = h.el('g', { 'clip-path': `url(#${clipId})` }, svg);
          const state = { a: 0.3, b: 2.5, mode: 'mse' };
          const out = h.readout('');
          let sa, sb;
          const render = () => {
            while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
            BASE.forEach(([x, y]) => {
              const yh = state.a * x + state.b, r = y - yh;
              if (state.mode === 'mse') {
                const s = Math.abs(r) * unit;
                h.el('rect', { x: sx(x), y: r >= 0 ? sy(y) : sy(yh), width: s, height: s, fill: 'var(--accent)', opacity: 0.22, stroke: 'var(--accent)' }, dyn);
              }
              h.el('line', { x1: sx(x), y1: sy(y), x2: sx(x), y2: sy(yh), stroke: 'var(--accent-2)', 'stroke-width': state.mode === 'mae' ? 6 : 2, opacity: state.mode === 'mae' ? 0.55 : 1 }, dyn);
            });
            h.el('line', { x1: sx(0), y1: sy(state.b), x2: sx(10), y2: sy(10 * state.a + state.b), stroke: 'var(--ink)', 'stroke-width': 2.5 }, dyn);
            BASE.forEach(([x, y]) => h.el('circle', { cx: sx(x), cy: sy(y), r: 6, fill: 'var(--pos)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, dyn));
            const m = mse(BASE, state.a, state.b), a = mae(BASE, state.a, state.b);
            out(state.mode === 'mse'
              ? `<b>MSE = ${h.fmt(m, 3)}</b> &nbsp; MAE = ${h.fmt(a, 3)}`
              : `MSE = ${h.fmt(m, 3)} &nbsp; <b>MAE = ${h.fmt(a, 3)}</b>`);
          };
          sa = h.slider({ label: 'pente a', min: -1, max: 2, step: 0.01, value: state.a, decimals: 2, onInput: (v) => { state.a = v; render(); } });
          sb = h.slider({ label: 'ordonnée b', min: -3, max: 8, step: 0.05, value: state.b, decimals: 2, onInput: (v) => { state.b = v; render(); } });
          h.toggle([['mse', 'Carrés (MSE)'], ['mae', 'Longueurs (MAE)']], state.mode, (v) => { state.mode = v; render(); });
          let seq = 0;
          const go = async (t) => {
            const id = ++seq, a0 = state.a, b0 = state.b;
            await h.tween(700, (p) => {
              if (id !== seq) return;
              state.a = h.lerp(a0, t.a, p);
              state.b = h.lerp(b0, t.b, p);
              sa.set(state.a);
              sb.set(state.b);
              render();
            });
          };
          h.button('Ajuster (MSE)', () => go(fitMSE(BASE)));
          h.button('Ajuster (MAE)', () => go(fitMAE(BASE)));
          render();
        }
      },
      {
        title: 'Effet d\'un outlier',
        duration: 12000,
        text: String.raw`<p>Un <b>outlier</b> est un point aberrant (erreur de mesure, cas rare). Voyons son effet sur la meilleure droite pour chaque coût.</p>
<p>Avec la MSE, l'écart de l'outlier est <b>élevé au carré</b> : il domine la somme et la droite est tirée vers lui. La MAE est plus <b>robuste</b> : l'outlier pèse seulement proportionnellement à son écart.</p>
<p>Activez l'outlier puis déplacez-le verticalement (curseur ou glisser le point rouge).</p>`,
        note: String.raw`MSE : sensible aux outliers, lisse partout (pratique pour le gradient). MAE : robuste, mais non dérivable en 0.`,
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x0: 60, x1: 690, y0: 20, y1: 350 };
          const sx = h.scale(0, 10, b.x0, b.x1), sy = h.scale(0, 10, b.y1, b.y0);
          axes(h, svg, b, sx, sy, { xt: [0, 2, 4, 6, 8, 10], yt: [0, 2, 4, 6, 8, 10], grid: true, xl: 'x', yl: 'y' });
          const state = { on: false, y: 9.5 };
          const cur = { ma: 0, mb: 0, aa: 0, ab: 0 };
          const dyn = h.el('g', {}, svg);
          const out = h.readout('');
          let seq = 0;
          const points = () => (state.on ? BASE.concat([[9, state.y]]) : BASE);
          const targets = () => {
            const pts = points(), f = fitMSE(pts), g = fitMAE(pts);
            return { ma: f.a, mb: f.b, aa: g.a, ab: g.b };
          };
          const render = () => {
            while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
            const line = (a, c, col, w) => h.el('line', { x1: sx(0), y1: sy(c), x2: sx(10), y2: sy(10 * a + c), stroke: col, 'stroke-width': w }, dyn);
            line(cur.ma, cur.mb, 'var(--accent)', 3);
            line(cur.aa, cur.ab, 'var(--accent-2)', 3);
            BASE.forEach(([x, y]) => h.el('circle', { cx: sx(x), cy: sy(y), r: 6, fill: 'var(--pos)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, dyn));
            if (state.on) {
              h.el('circle', { cx: sx(9), cy: sy(state.y), r: 8, fill: 'var(--bad)', stroke: 'var(--ink)', 'stroke-width': 1.5, style: 'cursor:ns-resize' }, dyn);
              txt(h, dyn, sx(9) - 14, sy(state.y), 'outlier', { size: 12, anchor: 'end', mid: true, fill: 'var(--bad)' });
            }
            h.el('line', { x1: b.x0 + 14, y1: b.y0 + 20, x2: b.x0 + 44, y2: b.y0 + 20, stroke: 'var(--accent)', 'stroke-width': 3 }, dyn);
            txt(h, dyn, b.x0 + 52, b.y0 + 20, 'Droite optimale MSE', { size: 13, anchor: 'start', mid: true });
            h.el('line', { x1: b.x0 + 14, y1: b.y0 + 42, x2: b.x0 + 44, y2: b.y0 + 42, stroke: 'var(--accent-2)', 'stroke-width': 3 }, dyn);
            txt(h, dyn, b.x0 + 52, b.y0 + 42, 'Droite optimale MAE', { size: 13, anchor: 'start', mid: true });
            out(`pente MSE = <b>${h.fmt(cur.ma, 2)}</b> &nbsp; pente MAE = <b>${h.fmt(cur.aa, 2)}</b>`);
          };
          const update = async (animate) => {
            const t = targets(), id = ++seq, f = Object.assign({}, cur);
            if (!animate) { Object.assign(cur, t); render(); return; }
            await h.tween(500, (p) => {
              if (id !== seq) return;
              for (const k in t) cur[k] = h.lerp(f[k], t[k], p);
              render();
            });
          };
          const wrap = h.toggle([['no', 'Sans outlier'], ['yes', 'Avec outlier']], 'no', (v) => { state.on = v === 'yes'; update(true); });
          const sl = h.slider({ label: "y de l'outlier", min: 0, max: 10, step: 0.1, value: state.y, decimals: 1, onInput: (v) => { state.y = v; update(false); } });
          let drag = false;
          const move = (ev) => {
            state.y = Math.max(0, Math.min(10, sy.invert(svgPoint(svg, ev).y)));
            sl.set(state.y);
            update(false);
          };
          svg.addEventListener('pointerdown', (ev) => {
            if (!state.on) return;
            const p = svgPoint(svg, ev);
            if (Math.hypot(p.x - sx(9), p.y - sy(state.y)) < 22) { drag = true; move(ev); }
          });
          const onMove = (ev) => { if (drag) move(ev); };
          const onUp = () => { drag = false; };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
          h.onLeave(() => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); });
          update(false);
          h.after(1600, () => pick(wrap, 1));
        }
      },
      {
        title: 'Entropie croisée binaire',
        duration: 12000,
        text: String.raw`<p>En classification binaire, le réseau sort une probabilité $p = P(y=1)$ (sigmoïde). L'<b>entropie croisée binaire</b> mesure combien de « surprise » la vraie étiquette $y\in\{0,1\}$ cause au modèle :</p>
$$L = -\big[\,y\ln p + (1-y)\ln(1-p)\,\big]$$
<p>Si $y=1$, la perte est $-\ln p$ ; si $y=0$, elle vaut $-\ln(1-p)$.</p>
<p>Choisissez l'étiquette vraie, puis faites varier $p$. Une réponse <b>sûre d'elle et fausse</b> est punie très durement : la perte explose quand $p\to 0$ (pour $y=1$).</p>`,
        note: String.raw`$-\ln(0{,}9)\approx 0{,}11$, $-\ln(0{,}5)\approx 0{,}69$, $-\ln(0{,}01)\approx 4{,}6$ : se tromper avec 99 % de confiance coûte 40 fois plus qu'une réponse « indécise ».`,
        check: {
          q: 'L\'étiquette vraie est $y=1$ et le modèle prédit $p=0{,}01$. La perte vaut environ :',
          choices: ['0,01', '0,69', '4,6'],
          answer: 2,
          explain: String.raw`$-\ln(0{,}01) = \ln 100 \approx 4{,}6$ : le modèle est très sûr de lui et se trompe.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x0: 70, x1: 690, y0: 30, y1: 350 };
          const sx = h.scale(0, 1, b.x0, b.x1), sy = h.scale(0, 5, b.y1, b.y0);
          axes(h, svg, b, sx, sy, { xt: [0, 0.25, 0.5, 0.75, 1], yt: [0, 1, 2, 3, 4, 5], grid: true, xl: 'probabilité prédite p', yl: 'perte L' });
          const dyn = h.el('g', {}, svg);
          const state = { y: 1, p: 0.1 };
          const out = h.readout('');
          const f1 = (p) => -Math.log(p), f0 = (p) => -Math.log(1 - p);
          const path = (f) => {
            let d = '';
            for (let i = 0; i <= 200; i++) {
              const p = 0.007 + 0.986 * i / 200;
              d += (i ? 'L' : 'M') + sx(p).toFixed(1) + ',' + sy(Math.min(5, f(p))).toFixed(1);
            }
            return d;
          };
          const render = () => {
            while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
            const sel1 = state.y === 1;
            h.el('path', { d: path(f1), fill: 'none', stroke: 'var(--pos)', 'stroke-width': sel1 ? 4 : 2, opacity: sel1 ? 1 : 0.3 }, dyn);
            h.el('path', { d: path(f0), fill: 'none', stroke: 'var(--neg)', 'stroke-width': sel1 ? 2 : 4, opacity: sel1 ? 0.3 : 1 }, dyn);
            txt(h, dyn, sx(0.3) + 8, sy(1.2) - 12, 'y = 1 : −ln(p)', { size: 13, anchor: 'start', fill: 'var(--pos)', bold: sel1 });
            txt(h, dyn, sx(0.7) - 8, sy(1.2) - 12, 'y = 0 : −ln(1 − p)', { size: 13, anchor: 'end', fill: 'var(--neg)', bold: !sel1 });
            const L = sel1 ? f1(state.p) : f0(state.p);
            const col = sel1 ? 'var(--pos)' : 'var(--neg)';
            h.el('line', { x1: sx(state.p), y1: b.y1, x2: sx(state.p), y2: sy(Math.min(5, L)), stroke: col, 'stroke-dasharray': '3 4' }, dyn);
            h.el('circle', { cx: sx(state.p), cy: sy(Math.min(5, L)), r: 8, fill: col, stroke: 'var(--surface)', 'stroke-width': 2 }, dyn);
            txt(h, dyn, 380, b.y0 + 34, 'perte = ' + h.fmt(L, 3), { size: 20, bold: true, mono: true });
            const verdict = L > 2.3 ? ['Confiant et faux : pénalité énorme', 'var(--bad)'] : L < 0.36 ? ['Confiant et juste : perte quasi nulle', 'var(--ok)'] : ['Indécis : perte modérée', 'var(--warn)'];
            txt(h, dyn, 380, b.y0 + 62, verdict[0], { size: 14, fill: verdict[1], bold: true });
            out(`$y=${state.y}$ &nbsp; $p=${h.fmt(state.p, 2)}$ &nbsp; $L=${h.fmt(L, 3)}$`);
          };
          const sl = h.slider({ label: 'p', min: 0.01, max: 0.99, step: 0.01, value: state.p, decimals: 2, onInput: (v) => { state.p = v; render(); } });
          h.toggle([[1, 'Vraie classe y = 1'], [0, 'Vraie classe y = 0']], state.y, (v) => { state.y = v; render(); });
          const hit = h.el('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: 'transparent', style: 'cursor:ew-resize' }, svg);
          let drag = false;
          const move = (ev) => {
            state.p = Math.max(0.01, Math.min(0.99, sx.invert(svgPoint(svg, ev).x)));
            sl.set(state.p);
            render();
          };
          hit.addEventListener('pointerdown', (ev) => { drag = true; move(ev); });
          const onMove = (ev) => { if (drag) move(ev); };
          const onUp = () => { drag = false; };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
          h.onLeave(() => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); });
          render();
          (async () => {
            await h.sleep(500);
            await h.tween(1800, (q) => { state.p = h.lerp(0.1, 0.9, q); sl.set(state.p); render(); });
            await h.tween(900, (q) => { state.p = h.lerp(0.9, 0.1, q); sl.set(state.p); render(); });
          })();
        }
      },
      {
        title: 'Entropie croisée & softmax',
        duration: 14000,
        text: String.raw`<p>Avec $K$ classes exclusives, la dernière couche produit $K$ scores bruts, les <b>logits</b> $z_k$. La <b>softmax</b> les transforme en probabilités positives qui somment à 1 :</p>
$$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$
<p>L'<b>entropie croisée catégorielle</b> ne regarde que la probabilité de la vraie classe $c$ :</p>
$$L = -\ln p_c$$
<p>Bougez les logits : monter celui de la bonne classe fait baisser la perte, monter celui d'une mauvaise classe la fait grimper.</p>`,
        note: String.raw`La softmax est <b>relative</b> : seule la différence entre logits compte. Ajouter la même constante à tous les logits ne change pas les probabilités.`,
        check: {
          q: 'Trois classes, logits tous égaux. Quelle est la perte si la vraie classe est la 1 ?',
          choices: [String.raw`$-\ln(1/3)\approx 1{,}10$`, '0', String.raw`$-\ln(3)\approx -1{,}10$`],
          answer: 0,
          explain: String.raw`Logits égaux donnent $p_k=1/3$ pour chaque classe, donc $L=-\ln(1/3)=\ln 3\approx 1{,}10$.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const K = 3;
          const names = ['1', '2', '3'];
          const z = [2, 1, -1];
          const state = { c: 0 };
          const bz = { x0: 60, x1: 300, y0: 70, y1: 300 }, bp = { x0: 440, x1: 680, y0: 70, y1: 300 };
          const sz = h.scale(-4, 6, bz.y1, bz.y0), sp = h.scale(0, 1, bp.y1, bp.y0);
          const softmax = (v) => { const m = Math.max(...v); const e = v.map((a) => Math.exp(a - m)); const s = e.reduce((a, c) => a + c, 0); return e.map((a) => a / s); };
          txt(h, svg, (bz.x0 + bz.x1) / 2, 44, 'Logits z (scores bruts)', { size: 14, bold: true });
          txt(h, svg, (bp.x0 + bp.x1) / 2, 44, 'Probabilités p (softmax)', { size: 14, bold: true });
          [[bz, sz, [-4, -2, 0, 2, 4, 6]], [bp, sp, [0, 0.25, 0.5, 0.75, 1]]].forEach(([bx, s, yt]) => {
            yt.forEach((v) => {
              h.el('line', { x1: bx.x0, y1: s(v), x2: bx.x1, y2: s(v), stroke: 'var(--line)' }, svg);
              txt(h, svg, bx.x0 - 8, s(v), tick(v), { size: 12, fill: 'var(--muted)', anchor: 'end', mid: true });
            });
            h.el('line', { x1: bx.x0, y1: bx.y0, x2: bx.x0, y2: bx.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          });
          h.el('line', { x1: bz.x0, y1: sz(0), x2: bz.x1, y2: sz(0), stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          h.el('line', { x1: bp.x0, y1: bp.y1, x2: bp.x1, y2: bp.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          h.el('line', { x1: 322, y1: 185, x2: 418, y2: 185, stroke: 'var(--accent)', 'stroke-width': 3, 'marker-end': 'url(#arrow-accent)' }, svg);
          txt(h, svg, 370, 168, 'softmax', { size: 14, fill: 'var(--accent)', bold: true });
          const cw = (bz.x1 - bz.x0) / K;
          const zBars = [], pBars = [], zLab = [], pLab = [], frames = [];
          for (let k = 0; k < K; k++) {
            const xc = bz.x0 + cw * (k + 0.5), xp = bp.x0 + cw * (k + 0.5);
            zBars.push(h.el('rect', { x: xc - 24, y: sz(0), width: 48, height: 0, rx: 3, fill: 'var(--accent-2)' }, svg));
            pBars.push(h.el('rect', { x: xp - 24, y: bp.y1, width: 48, height: 0, rx: 3, fill: 'var(--accent)' }, svg));
            zLab.push(txt(h, svg, xc, sz(0) - 8, '', { size: 13, mono: true }));
            pLab.push(txt(h, svg, xp, bp.y1 - 8, '', { size: 13, mono: true }));
            frames.push(h.el('rect', { x: xp - 29, y: bp.y0 - 4, width: 58, height: bp.y1 - bp.y0 + 8, rx: 6, fill: 'none', stroke: 'var(--ok)', 'stroke-width': 2.5, 'stroke-dasharray': '6 4', opacity: 0 }, svg));
            txt(h, svg, xc, bz.y1 + 22, 'classe ' + names[k], { size: 12, fill: 'var(--muted)' });
            txt(h, svg, xp, bp.y1 + 22, 'classe ' + names[k], { size: 12, fill: 'var(--muted)' });
          }
          const lossTxt = txt(h, svg, 360, 372, '', { size: 18, bold: true, mono: true });
          const lossSub = txt(h, svg, 360, 398, '', { size: 13, fill: 'var(--muted)' });
          const out = h.readout('');
          const cz = [0, 0, 0], cp = [0, 0, 0];
          const paint = () => {
            for (let k = 0; k < K; k++) {
              const y = sz(cz[k]), y0 = sz(0);
              zBars[k].setAttribute('y', Math.min(y, y0));
              zBars[k].setAttribute('height', Math.abs(y0 - y));
              zLab[k].setAttribute('y', cz[k] >= 0 ? y - 9 : y + 16);
              zLab[k].textContent = h.fmt(cz[k], 1);
              const py = sp(cp[k]);
              pBars[k].setAttribute('y', py);
              pBars[k].setAttribute('height', bp.y1 - py);
              pLab[k].setAttribute('y', py - 8);
              pLab[k].textContent = h.fmt(cp[k], 2);
            }
          };
          const info = () => {
            const p = softmax(z);
            const e = z.map((a) => Math.exp(a));
            const L = -Math.log(p[state.c]);
            frames.forEach((f, k) => f.setAttribute('opacity', k === state.c ? 1 : 0));
            lossTxt.textContent = `perte = −ln p${'₁₂₃'[state.c]} = −ln(${h.fmt(p[state.c], 3)}) = ${h.fmt(L, 3)}`;
            lossSub.textContent = 'La classe vraie est la classe ' + names[state.c] + ' (cadre vert).';
            out(`$e^{z}$ = ${e.map((a) => h.fmt(a, 2)).join(' ; ')} &nbsp; somme = ${h.fmt(e.reduce((a, c) => a + c, 0), 2)}`);
            return p;
          };
          let seq = 0;
          const update = async () => {
            const p = info(), id = ++seq, z0 = cz.slice(), p0 = cp.slice();
            await h.tween(350, (q) => {
              if (id !== seq) return;
              for (let k = 0; k < K; k++) { cz[k] = h.lerp(z0[k], z[k], q); cp[k] = h.lerp(p0[k], p[k], q); }
              paint();
            });
          };
          for (let k = 0; k < K; k++) {
            h.slider({ label: 'z' + '₁₂₃'[k], min: -4, max: 6, step: 0.1, value: z[k], decimals: 1, onInput: (v) => { z[k] = v; update(); } });
          }
          h.toggle([[0, 'Vraie : 1'], [1, 'Vraie : 2'], [2, 'Vraie : 3']], state.c, (v) => { state.c = v; info(); });
          paint();
          info();
          (async () => {
            const id = ++seq;
            await h.sleep(300);
            await h.tween(800, (q) => { if (id === seq) { for (let k = 0; k < K; k++) cz[k] = z[k] * q; paint(); } });
            await h.sleep(300);
            const p = softmax(z);
            await h.tween(900, (q) => { if (id === seq) { for (let k = 0; k < K; k++) cp[k] = p[k] * q; paint(); } });
          })();
        }
      },
      {
        title: 'Attention à l\'activation !',
        duration: 14000,
        text: String.raw`<p>La fonction de coût et l'<b>activation de la dernière couche</b> forment un couple : l'une suppose que la sortie a une certaine forme que l'autre doit produire.</p>
<ul>
<li><b>Linéaire</b> avec <b>MSE / MAE</b> (valeur réelle quelconque) ;</li>
<li><b>Sigmoïde</b> avec <b>entropie croisée binaire</b> (une probabilité) ;</li>
<li><b>Softmax</b> avec <b>entropie croisée catégorielle</b> ($K$ probabilités qui somment à 1).</li>
</ul>
<p>Cliquez sur une case du tableau pour voir pourquoi un autre mélange pose problème, ou choisissez un type de problème.</p>`,
        note: String.raw`Règle de pouce : <b>softmax pour la cross-entropy</b> catégorielle, sigmoïde pour la cross-entropy binaire, linéaire pour la MSE. Un mauvais couple donne des NaN ou un apprentissage inutile.`,
        check: {
          q: 'Classification en 5 classes exclusives avec entropie croisée catégorielle. Quelle activation sur la dernière couche ?',
          choices: ['Linéaire', 'Sigmoïde', 'Softmax'],
          answer: 2,
          explain: String.raw`La perte $-\ln p_c$ suppose de vraies probabilités qui somment à 1 : c'est le rôle de la softmax.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const acts = ['Linéaire', 'Sigmoïde', 'Softmax'];
          const losses = [['MSE / MAE', 'régression'], ['Entropie croisée', 'binaire'], ['Entropie croisée', 'catégorielle']];
          const R = [
            [['ok', 'Parfait pour une valeur réelle quelconque : la sortie $\\hat y = z$ n\'est pas bornée et la MSE compare directement $\\hat y$ à $y$.'],
              ['bad', 'La BCE calcule $\\ln p$ et $\\ln(1-p)$ : avec une sortie linéaire, $p$ peut être négatif ou supérieur à 1 et le logarithme n\'est pas défini (NaN).'],
              ['bad', 'La perte $-\\ln p_c$ exige de vraies probabilités. Des scores linéaires bruts ne somment pas à 1 et peuvent être négatifs.']],
            [['warn', 'Possible mais déconseillé : la sortie est bornée à $]0,1[$ (inadaptée à une régression) et le gradient devient quasi nul quand la sigmoïde sature.'],
              ['ok', 'Le bon couple pour la classification binaire : $p=\\sigma(z)\\in]0,1[$ et $L=-[y\\ln p+(1-y)\\ln(1-p)]$.'],
              ['bad', 'Les sorties sigmoïdes sont indépendantes et ne somment pas à 1 : ce ne sont pas des probabilités de classes exclusives.']],
            [['warn', 'Mal adapté : la sortie est une distribution (somme 1), pas une valeur réelle, et on perd la belle simplification gradient softmax + entropie croisée.'],
              ['bad', 'Redondant : avec un seul neurone, la softmax vaut toujours 1. Pour deux classes, utilisez une sigmoïde (ou deux neurones softmax avec l\'entropie catégorielle).'],
              ['ok', 'Le bon couple pour K classes exclusives : $p_k=e^{z_k}/\\sum_j e^{z_j}$ et $L=-\\ln p_c$.']]
          ];
          const style = { ok: ['var(--ok)', '✓ à utiliser'], warn: ['var(--warn)', '⚠ déconseillé'], bad: ['var(--bad)', '✗ à éviter'] };
          const gx = 190, gy = 96, cw = 150, rh = 60;
          losses.forEach((l, j) => {
            txt(h, svg, gx + cw * (j + 0.5), 52, l[0], { size: 13, bold: true });
            txt(h, svg, gx + cw * (j + 0.5), 72, l[1], { size: 12, fill: 'var(--muted)' });
          });
          txt(h, svg, 90, 80, 'Dernière couche', { size: 12, fill: 'var(--muted)' });
          const rects = [];
          acts.forEach((a, i) => {
            txt(h, svg, gx - 14, gy + rh * (i + 0.5), a, { size: 14, bold: true, anchor: 'end', mid: true });
            losses.forEach((l, j) => {
              const kind = R[i][j][0];
              const g = h.el('g', { style: 'cursor:pointer' }, svg);
              const r = h.el('rect', { x: gx + cw * j + 3, y: gy + rh * i + 3, width: cw - 6, height: rh - 6, rx: 8, fill: style[kind][0], 'fill-opacity': kind === 'ok' ? 0.28 : 0.13, stroke: style[kind][0], 'stroke-width': 1.5 }, g);
              txt(h, g, gx + cw * (j + 0.5), gy + rh * (i + 0.5), style[kind][1], { size: 13, mid: true, bold: kind === 'ok', fill: style[kind][0] });
              rects.push({ r, i, j });
              g.addEventListener('click', () => select(i, j));
            });
          });
          const box = h.el('g', {}, svg);
          const state = { i: 0, j: 0 };
          function select(i, j) {
            state.i = i;
            state.j = j;
            rects.forEach((o) => {
              const on = o.i === i && o.j === j;
              o.r.setAttribute('stroke', on ? 'var(--ink)' : style[R[o.i][o.j][0]][0]);
              o.r.setAttribute('stroke-width', on ? 3.5 : 1.5);
            });
            while (box.firstChild) box.removeChild(box.firstChild);
            const kind = R[i][j][0], col = style[kind][0];
            h.el('rect', { x: 40, y: 292, width: 640, height: 112, rx: 10, fill: 'var(--surface-2)', stroke: col, 'stroke-width': 2 }, box);
            h.label(box, 54, 298, 612, 100, `<div style="font-size:14px"><b style="color:${col}">${acts[i]} + ${losses[j][0]} (${losses[j][1]}) : ${style[kind][1]}</b></div><div style="font-size:14px;margin-top:6px">${R[i][j][1]}</div>`);
          }
          const wrap = h.toggle([[0, 'Régression'], [1, 'Binaire'], [2, 'Multiclasse']], 0, (v) => select(v, v));
          select(0, 0);
          h.after(2800, () => pick(wrap, 1));
          h.after(5600, () => pick(wrap, 2));
        }
      }
    ]
  });
})();
