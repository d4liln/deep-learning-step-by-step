/**
 * Lesson: gradient descent, learning rate, local minima, batch / minibatch / stochastic variants, momentum and learning-rate decay.
 */
(function () {
  let uid = 0;

  const B1 = { x: 70, y: 40, w: 600, h: 300 };
  const B2 = { x: 64, y: 40, w: 340, h: 272 };
  const LEVELS = [0.1, 0.3, 0.7, 1.5, 3, 5, 8, 12, 18];

  /**
   * Convex 1D loss used in the first steps, with its derivative.
   */
  const E1 = (w) => 0.5 * (w - 3) * (w - 3) + 0.5;
  const dE1 = (w) => w - 3;

  /**
   * Non-convex 1D loss with a global and a local minimum, with its derivative.
   */
  const F5 = (w) => 0.03 * Math.pow(w, 4) - 0.25 * w * w + 0.3 * w + 3;
  const dF5 = (w) => 0.12 * w * w * w - 0.5 * w + 0.3;

  const TOY = { x: [0.6, 1, 1.5, 2], y: [1, 2.4, 2.6, 4.6] };

  /**
   * Appends an SVG text element.
   */
  function txt(h, svg, x, y, s, o) {
    return h.el('text', Object.assign({ x, y, text: s, 'font-size': 12, fill: 'var(--muted)', 'text-anchor': 'middle' }, o || {}), svg);
  }

  /**
   * Creates a clipped group covering box b; the clip rectangle is exposed as g.rect.
   */
  function clipBox(h, svg, b) {
    const id = 'clip-gr' + (++uid);
    const cp = h.el('clipPath', { id }, svg);
    const rect = h.el('rect', { x: b.x, y: b.y, width: b.w, height: b.h }, cp);
    const g = h.el('g', { 'clip-path': 'url(#' + id + ')' }, svg);
    g.rect = rect;
    return g;
  }

  /**
   * Draws grid, ticks, axes and axis labels in box b and returns the scale functions.
   */
  function axes(h, svg, b, xr, yr, o) {
    const sx = h.scale(xr[0], xr[1], b.x, b.x + b.w);
    const sy = h.scale(yr[0], yr[1], b.y + b.h, b.y);
    const f = (v) => h.fmt(v, Number.isInteger(v) ? 0 : 1);
    o.xt.forEach((v) => {
      h.el('line', { x1: sx(v), y1: b.y, x2: sx(v), y2: b.y + b.h, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
      txt(h, svg, sx(v), b.y + b.h + 18, (o.fx || f)(v), { class: 'mono' });
    });
    o.yt.forEach((v) => {
      h.el('line', { x1: b.x, y1: sy(v), x2: b.x + b.w, y2: sy(v), stroke: 'var(--line)', 'stroke-width': 1 }, svg);
      txt(h, svg, b.x - 8, sy(v) + 4, (o.fy || f)(v), { class: 'mono', 'text-anchor': 'end' });
    });
    h.el('line', { x1: b.x, y1: b.y + b.h, x2: b.x + b.w, y2: b.y + b.h, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
    h.el('line', { x1: b.x, y1: b.y, x2: b.x, y2: b.y + b.h, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
    if (o.xl) h.label(svg, b.x + b.w / 2 - 90, b.y + b.h + 24, 180, 26, o.xl, 'center');
    if (o.yl) h.label(svg, b.x - 40, b.y - 36, 340, 26, o.yl);
    return { sx, sy };
  }

  /**
   * Builds an SVG path string for y = f(x) sampled on [x0, x1].
   */
  function pathOf(f, x0, x1, n, sx, sy) {
    let d = '';
    for (let k = 0; k <= n; k++) {
      const x = x0 + ((x1 - x0) * k) / n;
      d += (k ? 'L' : 'M') + sx(x).toFixed(1) + ' ' + sy(f(x)).toFixed(1);
    }
    return d;
  }

  /**
   * Makes the whole SVG draggable: onMove receives viewBox coordinates, onEnd is called on release.
   */
  function drag(svg, onMove, onEnd) {
    let on = false;
    svg.style.touchAction = 'none';
    svg.style.cursor = 'crosshair';
    const pos = (e) => {
      const p = svg.createSVGPoint();
      p.x = e.clientX;
      p.y = e.clientY;
      return p.matrixTransform(svg.getScreenCTM().inverse());
    };
    svg.addEventListener('pointerdown', (e) => {
      on = true;
      try { svg.setPointerCapture(e.pointerId); } catch (err) { on = true; }
      onMove(pos(e));
    });
    svg.addEventListener('pointermove', (e) => { if (on) onMove(pos(e)); });
    const end = () => { if (on) { on = false; if (onEnd) onEnd(); } };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
  }

  /**
   * Builds (once per lesson) a small noisy linear-regression dataset and the quadratic loss helpers.
   */
  function getData(h) {
    if (h.shared.reg) return h.shared.reg;
    const r = h.rng(2024);
    const n = 40;
    const xs = [];
    const ys = [];
    for (let i = 0; i < n; i++) {
      const x = -0.5 + 2.5 * r();
      xs.push(x);
      ys.push(0.6 + 1.1 * x + (r() + r() + r() - 1.5) * 0.9);
    }
    const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
    const mx = mean(xs);
    const mxx = mean(xs.map((x) => x * x));
    const my = mean(ys);
    const mxy = mean(xs.map((x, i) => x * ys[i]));
    const det = mxx - mx * mx;
    const star = [(my * mxx - mx * mxy) / det, (mxy - mx * my) / det];
    const loss = (w) => mean(xs.map((x, i) => Math.pow(w[0] + w[1] * x - ys[i], 2)));
    const grad = (w, idx) => {
      const ids = idx || xs.map((_, i) => i);
      let g0 = 0;
      let g1 = 0;
      ids.forEach((i) => {
        const e = w[0] + w[1] * xs[i] - ys[i];
        g0 += 2 * e;
        g1 += 2 * e * xs[i];
      });
      return [g0 / ids.length, g1 / ids.length];
    };
    const tr = 1 + mxx;
    const disc = Math.sqrt(Math.pow((1 - mxx) / 2, 2) + mx * mx);
    const l1 = tr / 2 + disc;
    const l2 = tr / 2 - disc;
    const th = 0.5 * Math.atan2(2 * mx, 1 - mxx);
    const ellipse = (c) => {
      const pts = [];
      for (let k = 0; k <= 90; k++) {
        const p = (k / 90) * 2 * Math.PI;
        const a = Math.sqrt(c / l1) * Math.cos(p);
        const b = Math.sqrt(c / l2) * Math.sin(p);
        pts.push([star[0] + a * Math.cos(th) - b * Math.sin(th), star[1] + a * Math.sin(th) + b * Math.cos(th)]);
      }
      return pts;
    };
    h.shared.reg = { n, xs, ys, star, emin: loss(star), loss, grad, ellipse };
    return h.shared.reg;
  }

  /**
   * Converts a list of parameter points to an SVG polyline path string.
   */
  function poly(pts, A) {
    return pts.map((p, i) => (i ? 'L' : 'M') + A.sx(p[0]).toFixed(1) + ' ' + A.sy(p[1]).toFixed(1)).join('');
  }

  /**
   * Draws axes and iso-loss ellipses of the regression loss; returns scales, clipped group and box.
   */
  function plot2D(h, svg, D) {
    const A = axes(h, svg, B2, [-2, 3], [-1, 3], { xt: [-2, -1, 0, 1, 2, 3], yt: [-1, 0, 1, 2, 3], xl: '$w_0$', yl: '$w_1$' });
    const g = clipBox(h, svg, B2);
    LEVELS.forEach((c) => {
      h.el('path', { d: poly(D.ellipse(c), A), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.2 }, g);
    });
    h.el('circle', { cx: A.sx(D.star[0]), cy: A.sy(D.star[1]), r: 4.5, fill: 'var(--ok)' }, g);
    return { A, g };
  }

  /**
   * Runs gradient descent on the regression loss with sampled minibatches of size m (m >= n means full batch).
   */
  function simulate(D, start, etaFn, m, T, r) {
    const all = D.xs.map((_, i) => i);
    const pts = [start.slice()];
    const bs = [];
    let w = start.slice();
    for (let t = 0; t < T; t++) {
      let idx = all;
      if (m < D.n) {
        const arr = all.slice();
        for (let k = 0; k < m; k++) {
          const j = k + Math.floor(r() * (D.n - k));
          const tmp = arr[k];
          arr[k] = arr[j];
          arr[j] = tmp;
        }
        idx = arr.slice(0, m);
      }
      const g = D.grad(w, idx);
      const eta = etaFn(t);
      w = [w[0] - eta * g[0], w[1] - eta * g[1]];
      pts.push(w);
      bs.push(idx);
    }
    return { pts, bs };
  }

  /**
   * Adds a coloured legend line (swatch + text) to an SVG.
   */
  function legend(h, svg, x, y, col, s, anchorW) {
    h.el('line', { x1: x, y1: y, x2: x + 22, y2: y, stroke: col, 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
    return txt(h, svg, x + 30, y + 4, s, { 'text-anchor': 'start', fill: 'var(--ink)', 'font-size': 13 });
  }

  /**
   * Heavy-ball descent on the narrow valley E(u, v) = 0.02 u^2 + v^2 (beta = 0 gives plain gradient descent).
   */
  function valley(beta, eta, T, start) {
    let w = start.slice();
    let v = [0, 0];
    const pts = [w.slice()];
    for (let t = 0; t < T; t++) {
      const g = [0.04 * w[0], 2 * w[1]];
      v = [beta * v[0] - eta * g[0], beta * v[1] - eta * g[1]];
      w = [w[0] + v[0], w[1] + v[1]];
      pts.push(w.slice());
    }
    return pts;
  }

  /**
   * Formats a power of ten label such as 10^-2 using unicode superscripts.
   */
  function pow10(v) {
    const sup = ['⁰', '¹', '²', '³'];
    return v < 0 ? '10⁻' + sup[-v] : '10' + sup[v];
  }

  Course.register({
    id: 'gradient',
    order: 5,
    title: 'Descente de gradient & SGD',
    summary: "Comment un réseau apprend : on <strong>descend</strong> la fonction de coût en suivant l'opposé du gradient, avec toutes les données (batch), un <em>minibatch</em> ou un seul exemple (SGD), puis on améliore avec le momentum et le decay.",
    pdfPages: '20-21, 26',
    steps: [
      {
        title: 'Entraîner = minimiser',
        text: String.raw`<p>Entraîner un MLP, c'est chercher les poids $w$ qui rendent l'<strong>erreur</strong> $E(w)$ la plus petite possible sur les exemples d'apprentissage.</p>
<p>Cette erreur est une <strong>somme de fonctions dérivables</strong> : chaque exemple $i$ apporte sa propre erreur $e_i(w)$.</p>
$$E(w)=\frac1n\sum_{i=1}^{n} e_i(w)$$
<p>Ici, un seul poids : chaque courbe fine est la contribution d'un exemple, la courbe épaisse est leur somme. Déplacez $w$ : l'erreur <em>dépend des poids</em>.</p>`,
        note: String.raw`Pour un vrai réseau, $E$ dépend de millions de poids et n'est <strong>pas convexe</strong> : on ne peut pas résoudre $E'(w)=0$ à la main, d'où l'approche itérative.`,
        draw(h) {
          const svg = h.svg(720, 420);
          const A = axes(h, svg, B1, [0, 4], [0, 9], { xt: [0, 1, 2, 3, 4], yt: [0, 3, 6, 9], xl: 'Poids $w$', yl: 'Erreur $E(w)$' });
          const cols = ['var(--pos)', 'var(--neg)', 'var(--accent-2)', 'var(--warn)'];
          const ei = (i, w) => Math.pow(w * TOY.x[i] - TOY.y[i], 2) / 4;
          const E = (w) => [0, 1, 2, 3].reduce((s, i) => s + ei(i, w), 0);
          const g = clipBox(h, svg, B1);
          const curves = cols.map((c, i) => h.el('path', {
            d: pathOf((w) => ei(i, w), 0, 4, 100, A.sx, A.sy), fill: 'none', stroke: c, 'stroke-width': 1.8, opacity: 0, style: 'transition: opacity .5s'
          }, g));
          const sumG = clipBox(h, svg, B1);
          sumG.rect.setAttribute('width', 0);
          h.el('path', { d: pathOf(E, 0, 4, 120, A.sx, A.sy), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 4 }, sumG);
          legend(h, svg, B1.x + 220, B1.y + 16, 'var(--muted)', 'une courbe fine = e_i(w)/n', 0);
          legend(h, svg, B1.x + 220, B1.y + 38, 'var(--accent)', 'courbe épaisse = E(w), la somme', 0);
          const vline = h.el('line', { y1: B1.y, y2: B1.y + B1.h, stroke: 'var(--ink)', 'stroke-dasharray': '4 3', 'stroke-width': 1.2 }, svg);
          const dots = cols.map((c) => h.el('circle', { r: 4, fill: c }, svg));
          const big = h.el('circle', { r: 8, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          const rd = h.readout('');
          const upd = (w) => {
            h.attr(vline, { x1: A.sx(w), x2: A.sx(w) });
            dots.forEach((d, i) => h.attr(d, { cx: A.sx(w), cy: A.sy(ei(i, w)) }));
            h.attr(big, { cx: A.sx(w), cy: A.sy(E(w)) });
            rd(`w = ${h.fmt(w)} &nbsp;·&nbsp; E(w) = ${h.fmt(E(w))}`);
          };
          const sl = h.slider({ label: 'poids $w$', min: 0, max: 4, step: 0.02, value: 0.8, onInput: upd });
          upd(0.8);
          (async () => {
            for (const c of curves) { c.setAttribute('opacity', 0.9); await h.sleep(300); }
            await h.tween(1200, (p) => sumG.rect.setAttribute('width', B1.w * p));
          })();
          sumG.rect.setAttribute('width', 0);
        }
      },
      {
        title: 'Pente et tangente',
        text: String.raw`<p>Comment savoir dans quel sens bouger $w$ ? La <strong>dérivée</strong> $E'(w)$ est la pente de la <strong>tangente</strong> à la courbe au point courant.</p>
<ul><li>$E'(w)>0$ : la courbe monte vers la droite, il faut aller vers la <strong>gauche</strong>.</li>
<li>$E'(w)<0$ : la courbe descend vers la droite, il faut aller vers la <strong>droite</strong>.</li></ul>
<p>Dans les deux cas, on va dans la direction de $-E'(w)$ : c'est la <em>direction opposée au gradient</em>. Faites glisser la bille.</p>`,
        note: String.raw`Plus la pente est forte, plus on est loin du fond : $|E'(w)|$ indique aussi « de combien » bouger.`,
        check: {
          q: String.raw`En un point où $E'(w)=-2$, dans quel sens faut-il déplacer $w$ pour diminuer l'erreur ?`,
          choices: ['Vers la gauche (w diminue)', 'Vers la droite (w augmente)', 'On ne bouge pas'],
          answer: 1,
          explain: String.raw`La pente est négative : la courbe descend quand $w$ augmente. On va dans le sens de $-E'(w)=+2$, donc vers la droite.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const A = axes(h, svg, B1, [-1, 7], [0, 9], { xt: [-1, 0, 1, 2, 3, 4, 5, 6, 7], yt: [0, 3, 6, 9], xl: '$w$', yl: '$E(w)$' });
          h.el('path', { d: pathOf(E1, -1, 7, 120, A.sx, A.sy), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3 }, svg);
          const g = clipBox(h, svg, B1);
          const tan = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 2.5 }, g);
          const ay = B1.y + B1.h - 16;
          const arr = h.el('line', { y1: ay, y2: ay, stroke: 'var(--accent-2)', 'stroke-width': 4, 'marker-end': 'url(#arrow-accent)' }, svg);
          const alab = txt(h, svg, 0, ay - 10, 'sens de descente', { fill: 'var(--accent-2)', 'font-size': 13 });
          const ball = h.el('circle', { r: 10, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1.5, style: 'cursor:grab' }, svg);
          const rd = h.readout('');
          let sl;
          const upd = (w) => {
            const e = E1(w);
            const d = dE1(w);
            h.attr(tan, { x1: A.sx(w - 2), y1: A.sy(e - 2 * d), x2: A.sx(w + 2), y2: A.sy(e + 2 * d) });
            h.attr(ball, { cx: A.sx(w), cy: A.sy(e) });
            const len = Math.min(110, 28 * Math.abs(d));
            const dir = d > 0 ? -1 : 1;
            const show = Math.abs(d) > 0.08;
            h.attr(arr, { x1: A.sx(w), x2: A.sx(w) + dir * len, opacity: show ? 1 : 0 });
            h.attr(alab, { x: A.sx(w) + dir * len / 2, opacity: show ? 1 : 0 });
            rd(`w = ${h.fmt(w)} &nbsp;·&nbsp; E′(w) = ${h.fmt(d)} &nbsp;→&nbsp; ${Math.abs(d) < 0.08 ? 'pente nulle : on est au fond' : d > 0 ? 'on va vers la gauche' : 'on va vers la droite'}`);
            sl.set(w);
          };
          sl = h.slider({ label: 'position $w$', min: -1, max: 7, step: 0.05, value: -0.5, onInput: upd });
          drag(svg, (p) => upd(Math.max(-1, Math.min(7, A.sx.invert(p.x)))));
          upd(-0.5);
        }
      },
      {
        title: 'Pas à pas : w ← w − η E′(w)',
        text: String.raw`<p>On répète la même idée : calculer la pente, faire un petit pas dans le sens opposé, recommencer. La règle de mise à jour est :</p>
$$w \leftarrow w - \eta\,E'(w)$$
<p>$\eta>0$ est le <strong>pas d'apprentissage</strong> (<em>learning rate</em>). À chaque itération, la tangente change et la bille se rapproche du fond ; les anciennes tangentes restent en gris.</p>`,
        note: String.raw`Près du minimum, $E'(w)\to 0$ : les pas deviennent naturellement de plus en plus petits.`,
        check: {
          q: String.raw`Avec $\eta=0{,}5$, $w=1$ et $E'(w)=-2$, quelle est la nouvelle valeur de $w$ ?`,
          choices: ['0', '2', '1,5'],
          answer: 1,
          explain: String.raw`$w \leftarrow 1 - 0{,}5\times(-2) = 2$ : on se déplace vers la droite, vers le minimum.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const A = axes(h, svg, B1, [-1, 7], [0, 9], { xt: [-1, 0, 1, 2, 3, 4, 5, 6, 7], yt: [0, 3, 6, 9], xl: '$w$', yl: '$E(w)$' });
          h.el('path', { d: pathOf(E1, -1, 7, 120, A.sx, A.sy), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3 }, svg);
          const g = clipBox(h, svg, B1);
          const olds = h.el('g', {}, g);
          const trail = h.el('g', {}, svg);
          const tan = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 2.5 }, g);
          const ball = h.el('circle', { r: 10, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          const rd = h.readout('');
          let w = -0.5;
          let t = 0;
          let busy = false;
          let auto = false;
          const place = (x) => {
            const e = E1(x);
            const d = dE1(x);
            h.attr(tan, { x1: A.sx(x - 2), y1: A.sy(e - 2 * d), x2: A.sx(x + 2), y2: A.sy(e + 2 * d) });
            h.attr(ball, { cx: A.sx(x), cy: A.sy(e) });
          };
          const info = () => rd(`t = ${t} &nbsp;·&nbsp; w = ${h.fmt(w)} &nbsp;·&nbsp; E(w) = ${h.fmt(E1(w))} &nbsp;·&nbsp; E′(w) = ${h.fmt(dE1(w))}`);
          const sl = h.slider({ label: 'pas $\\eta$', min: 0.05, max: 0.95, step: 0.05, value: 0.3, onInput: () => {} });
          const step = async () => {
            if (busy || t >= 40) return;
            busy = true;
            const x0 = w;
            const x1 = x0 - sl.get() * dE1(x0);
            h.el('line', { x1: A.sx(x0 - 2), y1: A.sy(E1(x0) - 2 * dE1(x0)), x2: A.sx(x0 + 2), y2: A.sy(E1(x0) + 2 * dE1(x0)), stroke: 'var(--muted)', 'stroke-width': 1.5, opacity: 0.45 }, olds);
            h.el('circle', { cx: A.sx(x0), cy: A.sy(E1(x0)), r: 4, fill: 'var(--accent)', opacity: 0.6 }, trail);
            await h.tween(550, (p) => {
              const x = h.lerp(x0, x1, p);
              h.attr(ball, { cx: A.sx(x), cy: A.sy(E1(x)) });
            });
            w = x1;
            t++;
            place(w);
            info();
            busy = false;
          };
          const reset = () => {
            auto = false;
            w = -0.5;
            t = 0;
            olds.innerHTML = '';
            trail.innerHTML = '';
            place(w);
            info();
          };
          h.button('Pas suivant', () => { auto = false; step(); }, 'primary');
          h.button('Lancer', async () => {
            auto = true;
            while (auto && h.alive() && t < 40 && Math.abs(dE1(w)) > 0.02) {
              await step();
              await h.sleep(150);
            }
            auto = false;
          });
          h.button('Réinitialiser', reset);
          reset();
        }
      },
      {
        title: "Le pas d'apprentissage η",
        text: String.raw`<p>Le choix de $\eta$ est crucial. Sur cette parabole, on montre facilement que $w_{t+1}-3=(1-\eta)(w_t-3)$ :</p>
<ul><li>$\eta$ <strong>trop petit</strong> : on avance, mais très lentement.</li>
<li>$\eta$ <strong>bien choisi</strong> : convergence rapide.</li>
<li>$\eta$ <strong>trop grand</strong> ($1<\eta<2$) : on dépasse le fond et on oscille ; au-delà de $2$, on <strong>diverge</strong>.</li></ul>
<p>Réglez $\eta$ puis lancez, ou comparez trois valeurs d'un coup.</p>`,
        note: String.raw`Le learning rate est le premier hyper-paramètre à régler : trop petit = lent, trop grand = instable.`,
        check: {
          q: String.raw`Sur cette parabole, que se passe-t-il avec $\eta=2{,}1$ ?`,
          choices: ['Convergence très rapide', 'Divergence : w s\'éloigne de plus en plus du minimum', 'w ne bouge pas'],
          answer: 1,
          explain: String.raw`Le facteur $1-\eta=-1{,}1$ a une valeur absolue supérieure à 1 : l'écart au minimum grandit à chaque pas, en alternant de signe.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const A = axes(h, svg, B1, [-1, 7], [0, 9], { xt: [-1, 0, 1, 2, 3, 4, 5, 6, 7], yt: [0, 3, 6, 9], xl: '$w$', yl: '$E(w)$' });
          h.el('path', { d: pathOf(E1, -1, 7, 120, A.sx, A.sy), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3 }, svg);
          const g = clipBox(h, svg, B1);
          const leg = h.el('g', {}, svg);
          const rd = h.readout('Choisissez η puis lancez.');
          let tok = 0;
          const regime = (e) => (e < 0.25 ? 'trop petit : très lent' : e <= 1 ? 'bien choisi : convergence sans oscillation' : e < 2 ? 'trop grand : oscillations autour du minimum' : 'divergence');
          const sl = h.slider({ label: 'pas $\\eta$', min: 0.05, max: 2.2, step: 0.05, value: 0.5, onInput: (v) => rd(`η = ${h.fmt(v)} → ${regime(v)}`) });
          const run = async (list) => {
            const my = ++tok;
            g.innerHTML = '';
            leg.innerHTML = '';
            const T = 20;
            const runs = list.map((r, i) => {
              const ws = [-0.5];
              for (let t = 0; t < T; t++) ws.push(ws[t] - r.eta * dE1(ws[t]));
              legend(h, leg, B1.x + 215, B1.y + 16 + 20 * i, r.col, `η = ${h.fmt(r.eta)}  (${r.name})`, 0);
              return {
                ws,
                col: r.col,
                line: h.el('path', { d: '', fill: 'none', stroke: r.col, 'stroke-width': 1.6, opacity: 0.8 }, g),
                dots: h.el('g', {}, g)
              };
            });
            const pt = (w) => A.sx(w).toFixed(1) + ' ' + A.sy(E1(w)).toFixed(1);
            for (let t = 0; t <= T; t++) {
              runs.forEach((r) => {
                const w = r.ws[t];
                if (!isFinite(w)) return;
                h.el('circle', { cx: A.sx(w), cy: A.sy(E1(w)), r: t === 0 ? 6 : 4.5, fill: r.col, stroke: 'var(--ink)', 'stroke-width': 1 }, r.dots);
                h.attr(r.line, { d: r.ws.slice(0, t + 1).map((x, k) => (k ? 'L' : 'M') + pt(x)).join('') });
              });
              rd(`itération ${t} / ${T}` + (runs.length === 1 ? ` · w = ${h.fmt(runs[0].ws[t])}` : ''));
              await h.sleep(260);
              if (my !== tok) return;
            }
            if (runs.length === 1) rd(`η = ${h.fmt(list[0].eta)} → ${regime(list[0].eta)}`);
            else rd('Comparaison : lent, rapide, oscillant.');
          };
          h.button('Lancer', () => run([{ eta: sl.get(), col: 'var(--accent)', name: 'choisi' }]), 'primary');
          h.button('Comparer 3 valeurs', () => run([
            { eta: 0.1, col: 'var(--warn)', name: 'trop petit' },
            { eta: 0.8, col: 'var(--ok)', name: 'bon' },
            { eta: 1.9, col: 'var(--bad)', name: 'trop grand' }
          ]));
          h.el('circle', { cx: A.sx(-0.5), cy: A.sy(E1(-0.5)), r: 6, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1 }, svg);
          h.after(300, () => run([
            { eta: 0.1, col: 'var(--warn)', name: 'trop petit' },
            { eta: 0.8, col: 'var(--ok)', name: 'bon' },
            { eta: 1.9, col: 'var(--bad)', name: 'trop grand' }
          ]));
        }
      },
      {
        title: 'Minima locaux',
        text: String.raw`<p>Quand la fonction de coût n'est <strong>pas convexe</strong> (c'est le cas d'un MLP), il existe plusieurs creux. La descente de gradient s'arrête dans celui où la bille <em>tombe</em>, et pas forcément dans le meilleur.</p>
<p>Faites glisser le point de départ : à gauche de la crête, on atteint le <strong>minimum global</strong> ; à droite, un <strong>minimum local</strong> plus haut.</p>
<p>L'algorithme est donc <strong>sensible à l'initialisation</strong> des poids.</p>`,
        note: 'La descente de gradient peut converger vers un minimum local : le résultat dépend du point de départ.',
        draw(h) {
          const svg = h.svg(720, 420);
          const A = axes(h, svg, B1, [-4, 4], [0, 9], { xt: [-4, -3, -2, -1, 0, 1, 2, 3, 4], yt: [0, 3, 6, 9], xl: '$w$', yl: '$E(w)$' });
          const crest = 0.68;
          h.el('rect', { x: B1.x, y: B1.y, width: A.sx(crest) - B1.x, height: B1.h, fill: 'var(--ok)', opacity: 0.08 }, svg);
          h.el('rect', { x: A.sx(crest), y: B1.y, width: B1.x + B1.w - A.sx(crest), height: B1.h, fill: 'var(--warn)', opacity: 0.1 }, svg);
          h.el('line', { x1: A.sx(crest), y1: B1.y, x2: A.sx(crest), y2: B1.y + B1.h, stroke: 'var(--muted)', 'stroke-dasharray': '5 4' }, svg);
          txt(h, svg, (B1.x + A.sx(crest)) / 2, B1.y + 20, 'bassin du minimum global', { fill: 'var(--ok)', 'font-size': 13 });
          txt(h, svg, (A.sx(crest) + B1.x + B1.w) / 2, B1.y + 20, 'bassin du minimum local', { fill: 'var(--warn)', 'font-size': 13 });
          h.el('path', { d: pathOf(F5, -4, 4, 160, A.sx, A.sy), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 3 }, svg);
          txt(h, svg, A.sx(-2.28), A.sy(F5(-2.28)) + 30, 'minimum global', { fill: 'var(--ok)', 'font-size': 13 });
          txt(h, svg, A.sx(1.62), A.sy(F5(1.62)) + 30, 'minimum local', { fill: 'var(--warn)', 'font-size': 13 });
          const g = clipBox(h, svg, B1);
          const trail = h.el('path', { d: '', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '2 3' }, g);
          const startM = h.el('circle', { r: 5, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2 }, svg);
          const ball = h.el('circle', { r: 10, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          const rd = h.readout('');
          let tok = 0;
          let w0 = 2.5;
          let sl;
          const put = (w) => h.attr(ball, { cx: A.sx(w), cy: A.sy(F5(w)) });
          const setStart = (w) => {
            tok++;
            w0 = Math.max(-4, Math.min(4, w));
            put(w0);
            h.attr(startM, { cx: A.sx(w0), cy: A.sy(F5(w0)) });
            h.attr(trail, { d: '' });
            sl.set(w0);
            rd(`départ w₀ = ${h.fmt(w0)}`);
          };
          const run = async () => {
            const my = ++tok;
            let w = w0;
            let d = 'M' + A.sx(w).toFixed(1) + ' ' + A.sy(F5(w)).toFixed(1);
            for (let t = 0; t < 120; t++) {
              const gr = dF5(w);
              if (Math.abs(gr) < 0.002) break;
              w -= 0.15 * gr;
              d += 'L' + A.sx(w).toFixed(1) + ' ' + A.sy(F5(w)).toFixed(1);
              put(w);
              h.attr(trail, { d });
              await h.sleep(40);
              if (my !== tok) return;
            }
            const glob = w < crest;
            rd(`arrêt en w = ${h.fmt(w)}, E = ${h.fmt(F5(w))} → ${glob ? 'minimum global' : 'minimum local (pas le meilleur !)'}`);
            h.attr(ball, { fill: glob ? 'var(--ok)' : 'var(--warn)' });
          };
          sl = h.slider({ label: 'départ $w_0$', min: -4, max: 4, step: 0.05, value: 2.5, onInput: (v) => { setStart(v); h.attr(ball, { fill: 'var(--accent)' }); } });
          drag(svg, (p) => { setStart(A.sx.invert(p.x)); h.attr(ball, { fill: 'var(--accent)' }); }, run);
          h.button('Lancer', run, 'primary');
          setStart(2.5);
          h.after(400, run);
        }
      },
      {
        title: 'Plusieurs paramètres : le gradient',
        text: String.raw`<p>Avec deux paramètres $(w_0,w_1)$, l'erreur est une surface. On la représente par des <strong>courbes de niveau</strong> : chaque ellipse relie les paramètres qui donnent la même erreur, le minimum est au centre.</p>
<p>On dérive par rapport à chaque paramètre (<strong>dérivées partielles</strong>) et on les range dans le <strong>gradient</strong> :</p>
$$\nabla E=\left(\frac{\partial E}{\partial w_0},\frac{\partial E}{\partial w_1}\right)$$
<p>Le gradient pointe vers la plus forte <em>montée</em>, perpendiculairement à la courbe de niveau. On descend en suivant $-\nabla E$. Faites glisser le point.</p>`,
        note: String.raw`« Suivre la direction opposée au gradient » : chaque composante de $-\nabla E$ indique comment bouger un paramètre.`,
        draw(h) {
          const D = getData(h);
          const svg = h.svg(720, 420);
          const P = plot2D(h, svg, D);
          const A = P.A;
          const g = P.g;
          const cur = h.el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.4 }, g);
          const tan = h.el('line', { stroke: 'var(--ink)', 'stroke-dasharray': '4 3', 'stroke-width': 1.3 }, g);
          const c0 = h.el('line', { stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'marker-end': 'url(#arrow-accent)' }, svg);
          const c1 = h.el('line', { stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'marker-end': 'url(#arrow-accent)' }, svg);
          const main = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 4, 'marker-end': 'url(#arrow-accent)' }, svg);
          const dot = h.el('circle', { r: 7, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
          const l0 = txt(h, svg, 0, 0, '−∂E/∂w₀', { fill: 'var(--accent-2)', 'font-size': 12 });
          const l1 = txt(h, svg, 0, 0, '−∂E/∂w₁', { fill: 'var(--accent-2)', 'font-size': 12, 'text-anchor': 'start' });
          const lm = txt(h, svg, 0, 0, '−∇E', { fill: 'var(--accent)', 'font-size': 14, 'font-weight': 700 });
          h.label(svg, 432, 50, 270, 300, `<div style="font-size:13px;line-height:1.45"><p><b>Courbes de niveau</b> : chaque ellipse regroupe les paramètres de même erreur $E$. Le point vert est le minimum.</p><p><b>Flèche épaisse</b> : $-\\nabla E$, la direction de plus forte descente. Elle est <b>perpendiculaire</b> à la courbe de niveau (tiretés = tangente).</p><p><b>Flèches fines</b> : les deux dérivées partielles, une par paramètre.</p></div>`);
          const rd = h.readout('');
          const upd = (w) => {
            const gr = D.grad(w);
            const mag = Math.hypot(gr[0], gr[1]) || 1e-9;
            const px = A.sx(w[0]);
            const py = A.sy(w[1]);
            const k = Math.min(34, 100 / mag);
            const sc = (v) => v * k;
            h.attr(dot, { cx: px, cy: py });
            h.attr(main, { x1: px, y1: py, x2: px - sc(gr[0]), y2: py + sc(gr[1]) });
            h.attr(c0, { x1: px, y1: py, x2: px - sc(gr[0]), y2: py, opacity: Math.abs(sc(gr[0])) > 6 ? 1 : 0 });
            h.attr(c1, { x1: px, y1: py, x2: px, y2: py + sc(gr[1]), opacity: Math.abs(sc(gr[1])) > 6 ? 1 : 0 });
            h.attr(l0, { x: px - sc(gr[0]) / 2, y: py + (gr[1] > 0 ? -8 : 16), opacity: Math.abs(sc(gr[0])) > 24 ? 1 : 0 });
            h.attr(l1, { x: px + 6, y: py + sc(gr[1]) / 2 + 4, opacity: Math.abs(sc(gr[1])) > 24 ? 1 : 0 });
            h.attr(lm, { x: px - sc(gr[0]) - (gr[0] > 0 ? 18 : -18), y: py + sc(gr[1]) + (gr[1] > 0 ? 16 : -8) });
            h.attr(cur, { d: poly(D.ellipse(Math.max(D.loss(w) - D.emin, 1e-3)), A) });
            h.attr(tan, { x1: px - (50 * gr[1]) / mag, y1: py - (50 * gr[0]) / mag, x2: px + (50 * gr[1]) / mag, y2: py + (50 * gr[0]) / mag });
            rd(`∂E/∂w₀ = ${h.fmt(gr[0])} &nbsp;·&nbsp; ∂E/∂w₁ = ${h.fmt(gr[1])} &nbsp;·&nbsp; E = ${h.fmt(D.loss(w))}`);
          };
          drag(svg, (p) => {
            upd([Math.max(-1.95, Math.min(2.95, A.sx.invert(p.x))), Math.max(-0.95, Math.min(2.95, A.sy.invert(p.y)))]);
          });
          upd([-1, 2]);
        }
      },
      {
        title: 'Algorithme (batch)',
        text: String.raw`<p>La <strong>descente de gradient</strong> est un algorithme itératif :</p>
<ol><li>initialiser les poids $w$ ;</li><li>calculer le gradient de l'erreur sur l'ensemble $S$ des exemples ;</li><li>mettre à jour $w \leftarrow w-\eta\,\nabla E(w)$ ;</li><li>recommencer jusqu'à convergence.</li></ol>
$$\nabla E(w)=\frac1{|S|}\sum_{i\in S}\nabla e_i(w),\qquad S=\text{tous les exemples}$$
<p>Ici $S$ est le jeu de données complet : c'est la version <strong>batch</strong>. Cliquez dans le plan pour choisir un autre point de départ.</p>`,
        note: String.raw`Chaque pas coupe les courbes de niveau à angle droit, d'où le chemin un peu en zigzag dans une vallée allongée.`,
        draw(h) {
          const D = getData(h);
          const svg = h.svg(720, 420);
          const P = plot2D(h, svg, D);
          const A = P.A;
          const g = P.g;
          const path = h.el('path', { d: '', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.2 }, g);
          const dots = h.el('g', {}, g);
          const startM = h.el('circle', { r: 6, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
          const ball = h.el('circle', { r: 7, fill: 'var(--accent)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          h.label(svg, 432, 50, 270, 300, `<div style="font-size:13px;line-height:1.45"><p><b>Chaque itération</b> utilise les $n=${D.n}$ exemples pour calculer le gradient exact.</p><p>Trajet <b>lisse</b> et déterministe : mêmes données, même départ, même chemin.</p><p>Un $\\eta$ trop grand fait osciller puis diverger (essayez 0,6).</p></div>`);
          const rd = h.readout('');
          let tok = 0;
          let start = [-1.6, -0.6];
          const sl = h.slider({ label: 'pas $\\eta$', min: 0.02, max: 0.6, step: 0.02, value: 0.12, onInput: () => {} });
          const toW = (p) => [Math.max(-1.95, Math.min(2.95, A.sx.invert(p.x))), Math.max(-0.95, Math.min(2.95, A.sy.invert(p.y)))];
          const mark = () => {
            h.attr(startM, { cx: A.sx(start[0]), cy: A.sy(start[1]) });
            h.attr(ball, { cx: A.sx(start[0]), cy: A.sy(start[1]) });
          };
          const run = async () => {
            const my = ++tok;
            const eta = sl.get();
            dots.innerHTML = '';
            let w = start.slice();
            let d = 'M' + A.sx(w[0]).toFixed(1) + ' ' + A.sy(w[1]).toFixed(1);
            mark();
            for (let t = 1; t <= 80; t++) {
              const gr = D.grad(w);
              w = [w[0] - eta * gr[0], w[1] - eta * gr[1]];
              const e = D.loss(w);
              if (!isFinite(e) || e > 1e4) { rd('Divergence : l\'erreur explose.'); return; }
              d += 'L' + A.sx(w[0]).toFixed(1) + ' ' + A.sy(w[1]).toFixed(1);
              h.attr(path, { d });
              h.el('circle', { cx: A.sx(w[0]), cy: A.sy(w[1]), r: 2.5, fill: 'var(--accent)' }, dots);
              h.attr(ball, { cx: A.sx(w[0]), cy: A.sy(w[1]) });
              rd(`itération ${t} &nbsp;·&nbsp; E = ${h.fmt(e, 3)} &nbsp;·&nbsp; ‖∇E‖ = ${h.fmt(Math.hypot(gr[0], gr[1]), 3)}`);
              await h.sleep(55);
              if (my !== tok) return;
            }
          };
          drag(svg, (p) => { tok++; start = toW(p); dots.innerHTML = ''; h.attr(path, { d: '' }); mark(); }, run);
          h.button('Lancer', run, 'primary');
          mark();
          h.after(350, run);
        }
      },
      {
        title: 'Batch, minibatch, stochastique',
        text: String.raw`<p>Comment apprendre avec $n$ exemples ? Calculer le gradient sur <strong>tous</strong> les exemples est souvent impossible quand $n$ est énorme.</p>
<ul><li><strong>Batch</strong> : les $n$ exemples à chaque pas.</li>
<li><strong>Minibatch</strong> : $m\ll n$ exemples tirés <em>au hasard</em>, à chaque pas.</li>
<li><strong>Stochastique (SGD)</strong> : minibatch de taille 1.</li></ul>
<p>Le gradient devient une estimation bruitée du vrai gradient, mais bien moins coûteuse. Les cases de droite s'allument pour les exemples tirés.</p>`,
        note: String.raw`Minibatch et SGD introduisent de l'aléatoire : le chemin est bruité, mais en moyenne il va dans la bonne direction.`,
        check: {
          q: 'Pourquoi le chemin du SGD (1 exemple par pas) est-il si irrégulier ?',
          choices: [
            'Parce que le gradient d\'un seul exemple est une estimation très bruitée du gradient total',
            'Parce que le pas η change à chaque itération',
            'Parce que le SGD n\'utilise pas de gradient'
          ],
          answer: 0,
          explain: "Chaque exemple « tire » les poids vers sa propre droite de régression : seule la moyenne sur beaucoup de pas pointe vers le minimum."
        },
        draw(h) {
          const D = getData(h);
          const svg = h.svg(720, 420);
          const P = plot2D(h, svg, D);
          const A = P.A;
          const g = P.g;
          const start = [-1.6, -0.6];
          const T = 80;
          const ETA = 0.1;
          const COLS = { batch: 'var(--accent)', mini: 'var(--accent-2)', sgd: 'var(--warn)' };
          let m = 8;
          let seed = 5;
          let tok = 0;
          let sims;
          const build = () => {
            sims = {
              batch: simulate(D, start, () => ETA, D.n, T, h.rng(seed)),
              mini: simulate(D, start, () => ETA, m, T, h.rng(seed + 1)),
              sgd: simulate(D, start, () => ETA, 1, T, h.rng(seed + 2))
            };
          };
          build();
          const lines = {};
          const balls = {};
          ['batch', 'mini', 'sgd'].forEach((k) => {
            lines[k] = h.el('path', { d: '', fill: 'none', stroke: COLS[k], 'stroke-width': k === 'batch' ? 2.6 : 1.6, 'stroke-linejoin': 'round', opacity: 0.95 }, g);
          });
          ['batch', 'mini', 'sgd'].forEach((k) => {
            balls[k] = h.el('circle', { r: 5.5, fill: COLS[k], stroke: 'var(--ink)', 'stroke-width': 1.2 }, g);
          });
          h.el('circle', { cx: A.sx(start[0]), cy: A.sy(start[1]), r: 6, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
          txt(h, svg, 432, 50, 'Les n = 40 exemples', { 'text-anchor': 'start', fill: 'var(--ink)', 'font-size': 13 });
          const cells = [];
          for (let i = 0; i < D.n; i++) {
            cells.push(h.el('rect', { x: 432 + (i % 10) * 26, y: 60 + Math.floor(i / 10) * 26, width: 22, height: 22, rx: 4, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.2 }, svg));
          }
          const l1 = legend(h, svg, 432, 196, COLS.batch, '', 0);
          const l2 = legend(h, svg, 432, 224, COLS.mini, '', 0);
          const l3 = legend(h, svg, 432, 252, COLS.sgd, '', 0);
          const setLeg = () => {
            l1.textContent = `Batch : ${D.n} exemples / pas`;
            l2.textContent = `Minibatch : m = ${m} exemples / pas`;
            l3.textContent = 'SGD : 1 exemple / pas';
          };
          setLeg();
          const cnt = txt(h, svg, 432, 292, '', { 'text-anchor': 'start', fill: 'var(--ink)', 'font-size': 13, class: 'mono' });
          txt(h, svg, 432, 316, 'case pleine = minibatch ; contour orange = SGD', { 'text-anchor': 'start', 'font-size': 11 });
          const show = (k) => {
            ['batch', 'mini', 'sgd'].forEach((n) => {
              const pts = sims[n].pts.slice(0, k + 1);
              h.attr(lines[n], { d: poly(pts, A) });
              const last = pts[pts.length - 1];
              h.attr(balls[n], { cx: A.sx(last[0]), cy: A.sy(last[1]) });
            });
            cells.forEach((c) => h.attr(c, { fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.2 }));
            if (k >= 1) {
              sims.mini.bs[k - 1].forEach((i) => h.attr(cells[i], { fill: COLS.mini }));
              h.attr(cells[sims.sgd.bs[k - 1][0]], { stroke: COLS.sgd, 'stroke-width': 4 });
            }
            cnt.textContent = `Itération ${k} / ${T}`;
          };
          const play = async () => {
            const my = ++tok;
            for (let k = 0; k <= T; k++) {
              show(k);
              await h.sleep(70);
              if (my !== tok) return;
            }
          };
          h.button('Rejouer', play, 'primary');
          h.button('Nouveau tirage', () => { seed += 7; build(); play(); });
          h.slider({ label: 'taille du minibatch $m$', min: 2, max: 20, step: 1, value: 8, decimals: 0, onInput: (v) => { m = v; setLeg(); build(); play(); } });
          show(0);
          h.after(300, play);
        }
      },
      {
        title: 'Hasard, régularisation, momentum',
        text: String.raw`<p>L'aléatoire du minibatch a des avantages : un <strong>effet de régularisation</strong> et souvent une meilleure convergence. Mais l'algorithme <strong>ne trouve pas toujours la meilleure solution</strong> et reste sensible à l'initialisation.</p>
<p>Plusieurs <strong>hyper-paramètres</strong> aident : le pas $\eta$, le <strong>momentum</strong> et le <em>decay</em>. Le momentum garde une « vitesse » :</p>
$$v \leftarrow \beta\,v-\eta\,\nabla E(w),\qquad w \leftarrow w+v$$
<p>Dans une vallée étroite, la descente simple zigzague en travers et avance lentement le long. Le momentum amortit les zigzags et accélère dans la bonne direction.</p>`,
        note: String.raw`$\beta=0$ redonne la descente simple. Un $\beta$ trop proche de 1 fait dépasser le minimum et osciller longtemps.`,
        check: {
          q: 'À quoi sert principalement le momentum ?',
          choices: [
            'À accumuler la direction des pas précédents pour accélérer et lisser la trajectoire',
            'À choisir les exemples du minibatch',
            'À empêcher le réseau d\'avoir des poids négatifs'
          ],
          answer: 0,
          explain: 'La vitesse v moyenne les gradients successifs : les composantes qui alternent (zigzag) s\'annulent, celles qui pointent toujours dans le même sens s\'additionnent.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x: 50, y: 44, w: 640, h: 320 };
          const A = axes(h, svg, b, [-4, 4], [-2, 2], { xt: [-4, -3, -2, -1, 0, 1, 2, 3, 4], yt: [-2, -1, 0, 1, 2], xl: '$u$ (direction plate)', yl: '$v$ (direction raide)' });
          const g = clipBox(h, svg, b);
          [0.04, 0.1, 0.25, 0.5, 1, 1.8, 2.6].forEach((c) => {
            h.el('ellipse', { cx: A.sx(0), cy: A.sy(0), rx: Math.sqrt(50 * c) * 80, ry: Math.sqrt(c) * 80, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.2 }, g);
          });
          h.el('circle', { cx: A.sx(0), cy: A.sy(0), r: 4.5, fill: 'var(--ok)' }, g);
          const start = [-3.5, 1.5];
          const ETA = 0.85;
          const T = 80;
          const gp = h.el('g', {}, g);
          const gm = h.el('g', {}, g);
          const lp = h.el('path', { fill: 'none', stroke: 'var(--warn)', 'stroke-width': 1.6 }, gp);
          const lm = h.el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 1.8 }, gm);
          const dp = h.el('g', {}, gp);
          const dm = h.el('g', {}, gm);
          legend(h, svg, b.x + 340, b.y + 18, 'var(--warn)', 'descente simple (β = 0)', 0);
          const ml = legend(h, svg, b.x + 340, b.y + 40, 'var(--accent)', '', 0);
          h.el('circle', { cx: A.sx(start[0]), cy: A.sy(start[1]), r: 6, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
          const rd = h.readout('');
          const En = (p) => 0.02 * p[0] * p[0] + p[1] * p[1];
          let tok = 0;
          let beta = 0.8;
          let plain = valley(0, ETA, T, start);
          let mom = valley(beta, ETA, T, start);
          const show = (k) => {
            h.attr(lp, { d: poly(plain.slice(0, k + 1), A) });
            h.attr(lm, { d: poly(mom.slice(0, k + 1), A) });
            dp.innerHTML = '';
            dm.innerHTML = '';
            plain.slice(0, k + 1).forEach((p) => h.el('circle', { cx: A.sx(p[0]), cy: A.sy(p[1]), r: 2.6, fill: 'var(--warn)' }, dp));
            mom.slice(0, k + 1).forEach((p) => h.el('circle', { cx: A.sx(p[0]), cy: A.sy(p[1]), r: 2.6, fill: 'var(--accent)' }, dm));
            rd(`itération ${k} &nbsp;·&nbsp; E simple = ${h.fmt(En(plain[k]), 3)} &nbsp;·&nbsp; E momentum = ${h.fmt(En(mom[k]), 3)}`);
          };
          const play = async () => {
            const my = ++tok;
            for (let k = 0; k <= T; k++) {
              show(k);
              await h.sleep(80);
              if (my !== tok) return;
            }
          };
          const setLeg = () => { ml.textContent = `avec momentum (β = ${h.fmt(beta)})`; };
          setLeg();
          h.slider({ label: 'momentum $\\beta$', min: 0, max: 0.95, step: 0.05, value: 0.8, onInput: (v) => { beta = v; mom = valley(beta, ETA, T, start); setLeg(); play(); } });
          h.button('Rejouer', play, 'primary');
          show(0);
          h.after(300, play);
        }
      },
      {
        title: 'Decay du pas & hyper-paramètres',
        text: String.raw`<p>Avec un pas $\eta$ constant, le SGD finit par <strong>tourner autour</strong> du minimum sans jamais s'y poser : chaque exemple tire encore les poids dans sa direction.</p>
<p>Le <strong>decay</strong> diminue le pas au fil des itérations, par exemple :</p>
$$\eta_t=\frac{\eta_0}{1+\lambda\,t}$$
<p>Au début on explore avec de grands pas, à la fin on affine avec de petits pas. Sur le graphique, l'erreur (échelle logarithmique) d'un SGD à 1 exemple par pas, avec et sans decay.</p>`,
        note: String.raw`<strong>Hyper-paramètres à régler</strong> : pas d'apprentissage $\eta$, momentum $\beta$, decay $\lambda$, taille de minibatch $m$. Ils se choisissent par essais sur un jeu de validation.`,
        check: {
          q: 'Pourquoi diminuer le pas d\'apprentissage au cours de l\'entraînement ?',
          choices: [
            'Pour converger plus finement autour du minimum malgré le bruit du SGD',
            'Pour éviter de calculer le gradient',
            'Pour que le réseau ait moins de paramètres'
          ],
          answer: 0,
          explain: 'Avec un petit pas final, les fluctuations dues au tirage d\'un seul exemple sont amorties et les poids se stabilisent près du minimum.'
        },
        draw(h) {
          const D = getData(h);
          const svg = h.svg(720, 420);
          const b = { x: 70, y: 44, w: 600, h: 290 };
          const A = axes(h, svg, b, [0, 300], [-3.5, 1.5], { xt: [0, 50, 100, 150, 200, 250, 300], yt: [-3, -2, -1, 0, 1], fy: pow10, xl: 'itération $t$', yl: 'Excès d\'erreur $E(w_t)-E_{\\min}$ (échelle log)' });
          const g = clipBox(h, svg, b);
          const ETA0 = 0.1;
          const T = 300;
          const start = [-1.6, -0.6];
          const curve = (lam) => {
            const sim = simulate(D, start, (t) => ETA0 / (1 + lam * t), 1, T, h.rng(77));
            return sim.pts.map((p, t) => 'LM'.charAt(t ? 0 : 1) + A.sx(t).toFixed(1) + ' ' + A.sy(Math.log10(Math.max(D.loss(p) - D.emin, 1e-6))).toFixed(1)).join('');
          };
          h.el('path', { d: curve(0), fill: 'none', stroke: 'var(--warn)', 'stroke-width': 1.6, 'stroke-linejoin': 'round' }, g);
          const dec = h.el('path', { d: curve(0.03), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.2, 'stroke-linejoin': 'round' }, g);
          legend(h, svg, b.x + b.w - 250, b.y + 20, 'var(--warn)', 'pas constant η = 0,10', 0);
          const dl = legend(h, svg, b.x + b.w - 250, b.y + 42, 'var(--accent)', '', 0);
          const rd = h.readout('');
          let lam = 0.03;
          const info = () => {
            dl.textContent = `avec decay λ = ${h.fmt(lam, 3)}`;
            rd(`η final = ${h.fmt(ETA0 / (1 + lam * T), 3)} (départ ${h.fmt(ETA0, 2)})`);
          };
          const play = async () => {
            g.rect.setAttribute('width', 0);
            await h.tween(3200, (p) => g.rect.setAttribute('width', b.w * p), h.ease.linear);
          };
          h.slider({ label: 'decay $\\lambda$', min: 0, max: 0.2, step: 0.005, value: 0.03, decimals: 3, onInput: (v) => { lam = v; h.attr(dec, { d: curve(lam) }); info(); } });
          h.button('Rejouer', play, 'primary');
          info();
          play();
        }
      }
    ]
  });
})();
