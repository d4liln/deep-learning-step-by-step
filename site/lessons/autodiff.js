/**
 * Lesson: gradient computation in practice (numerical, symbolic and automatic differentiation, computational graph, reverse mode, toolboxes, GPU, practical sessions).
 */
(function () {
  const F = (a, b) => Math.log(a) + a * b - Math.sin(b);
  const GRAD = (a, b) => [1 / a + b, a - Math.cos(b)];

  const NODES = [
    { id: 'x1', sym: 'x₁', x: 70, y: 120 },
    { id: 'x2', sym: 'x₂', x: 70, y: 320 },
    { id: 'ln', sym: 'ln', x: 250, y: 55 },
    { id: 'mul', sym: '×', x: 250, y: 215 },
    { id: 'sin', sym: 'sin', x: 250, y: 365 },
    { id: 'add', sym: '+', x: 430, y: 135 },
    { id: 'sub', sym: '−', x: 580, y: 250 }
  ];

  const EDGES = [
    { a: 'x1', b: 'ln', sym: '1/x₁', loc: (v) => 1 / v.x1 },
    { a: 'x1', b: 'mul', sym: 'x₂', loc: (v) => v.x2 },
    { a: 'x2', b: 'mul', sym: 'x₁', loc: (v) => v.x1 },
    { a: 'x2', b: 'sin', sym: 'cos x₂', loc: (v) => Math.cos(v.x2) },
    { a: 'ln', b: 'add', sym: '1', loc: () => 1 },
    { a: 'mul', b: 'add', sym: '1', loc: () => 1 },
    { a: 'add', b: 'sub', sym: '1', loc: () => 1 },
    { a: 'sin', b: 'sub', sym: '−1', loc: () => -1 }
  ];

  const NAMES = { x1: 'x₁', x2: 'x₂', ln: 'ln', mul: '×', sin: 'sin', add: '+', sub: '−' };

  /**
   * Evaluates the computational graph of f and runs the reverse accumulation of adjoints.
   */
  function evalGraph(x1, x2) {
    const v = { x1, x2 };
    v.ln = Math.log(x1);
    v.mul = x1 * x2;
    v.sin = Math.sin(x2);
    v.add = v.ln + v.mul;
    v.sub = v.add - v.sin;
    const loc = EDGES.map((e) => e.loc(v));
    const adj = {};
    NODES.forEach((n) => { adj[n.id] = 0; });
    adj.sub = 1;
    const contrib = [];
    for (let i = NODES.length - 1; i >= 0; i--) {
      const id = NODES[i].id;
      EDGES.forEach((e, k) => {
        if (e.b === id) {
          contrib[k] = adj[id] * loc[k];
          adj[e.a] += contrib[k];
        }
      });
    }
    return { v, loc, adj, contrib };
  }

  /**
   * Creates an SVG text element styled through inline CSS so that theme variables win over the default rules.
   */
  function T(h, parent, x, y, text, o) {
    const opt = o || {};
    const e = h.el('text', { x, y, text, 'text-anchor': opt.anchor || 'middle', class: opt.mono ? 'mono' : '' }, parent);
    let s = 'fill:' + (opt.fill || 'var(--ink)') + ';font-size:' + (opt.size || 13) + 'px;';
    if (opt.halo) s += 'paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round;';
    if (opt.weight) s += 'font-weight:' + opt.weight + ';';
    if (opt.op !== undefined) s += 'opacity:' + opt.op + ';transition:opacity .4s;';
    e.setAttribute('style', s);
    return e;
  }

  /**
   * Returns a colour-mix expression giving a pale tint of a theme colour.
   */
  function tint(c, p) {
    return 'color-mix(in srgb, var(--' + c + ') ' + (p || 25) + '%, var(--surface))';
  }

  /**
   * Draws the computational graph of f with node values, adjoints and local derivatives; returns a controller.
   */
  function makeGraph(h, svg, x1, x2, opts) {
    const o = opts || {};
    const R = 22;
    const pos = {};
    NODES.forEach((n) => { pos[n.id] = n; });
    const g = { nodes: {}, edges: [], mode: 'sym', data: null };
    const edgeLayer = h.el('g', {}, svg);
    const nodeLayer = h.el('g', {}, svg);
    const textLayer = h.el('g', {}, svg);
    const fxLayer = h.el('g', {}, svg);

    EDGES.forEach((e) => {
      const A = pos[e.a];
      const B = pos[e.b];
      const dx = B.x - A.x;
      const dy = B.y - A.y;
      const d = Math.hypot(dx, dy);
      const ux = dx / d;
      const uy = dy / d;
      const sx = A.x + ux * R;
      const sy = A.y + uy * R;
      const ex = B.x - ux * (R + 3);
      const ey = B.y - uy * (R + 3);
      const grp = h.el('g', { style: 'transition:opacity .5s' }, edgeLayer);
      h.el('line', { x1: sx, y1: sy, x2: ex, y2: ey, stroke: 'var(--muted)', 'stroke-width': 1.8, 'marker-end': 'url(#arrow-muted)' }, grp);
      const hl = h.el('line', { x1: sx, y1: sy, x2: ex, y2: ey, 'stroke-width': 4, 'stroke-linecap': 'round' }, grp);
      hl.style.stroke = 'var(--accent)';
      hl.style.opacity = 0;
      hl.style.transition = 'opacity .3s';
      const lab = T(h, textLayer, (sx + ex) / 2, (sy + ey) / 2 + 4, e.sym, { mono: true, size: 12, halo: true, op: 0 });
      g.edges.push({ e, grp, hl, lab, sx, sy, ex, ey });
    });

    NODES.forEach((n) => {
      const grp = h.el('g', { style: 'transition:opacity .5s' }, nodeLayer);
      const c = h.el('circle', { cx: n.x, cy: n.y, r: R, 'stroke-width': 1.5 }, grp);
      c.style.fill = 'var(--surface)';
      c.style.stroke = 'var(--ink)';
      c.style.transition = 'fill .3s, stroke .3s';
      T(h, grp, n.x, n.y + 5, n.sym, { size: n.sym.length > 2 ? 13 : 17, weight: 600 });
      const val = T(h, textLayer, n.x, n.y + R + 17, '', { mono: true, size: 13, fill: 'var(--accent)', halo: true, weight: 700, op: 0 });
      const adj = T(h, textLayer, n.x, n.y - R - 9, '', { mono: true, size: 13, fill: 'var(--accent-2)', halo: true, weight: 700, op: 0 });
      g.nodes[n.id] = { grp, c, val, adj };
    });

    const out = h.el('g', { style: 'transition:opacity .5s' }, nodeLayer);
    h.el('line', { x1: 580 + R + 3, y1: 250, x2: 652, y2: 250, stroke: 'var(--ink)', 'stroke-width': 1.8, 'marker-end': 'url(#arrow)' }, out);
    T(h, out, 674, 256, 'f', { size: 18, weight: 700 });
    g.out = out;

    if (o.legend) {
      T(h, svg, 705, 20, '● valeur du nœud', { anchor: 'end', fill: 'var(--accent)', size: 13, weight: 600 });
      if (o.legend === 'both') T(h, svg, 705, 40, '● ∂f / ∂nœud (adjoint)', { anchor: 'end', fill: 'var(--accent-2)', size: 13, weight: 600 });
    }

    g.vis = (el, on) => { el.style.opacity = on ? 1 : 0; };

    g.update = (a, b) => {
      g.x1 = a;
      g.x2 = b;
      g.data = evalGraph(a, b);
      NODES.forEach((n) => { g.nodes[n.id].val.textContent = h.fmt(g.data.v[n.id], 2); });
      g.edges.forEach((E, k) => { E.lab.textContent = g.mode === 'num' ? h.fmt(g.data.loc[k], 2) : E.e.sym; });
    };

    g.setMode = (m) => {
      g.mode = m;
      g.edges.forEach((E, k) => { E.lab.textContent = m === 'num' ? h.fmt(g.data.loc[k], 2) : E.e.sym; });
    };

    g.mark = (id, kind) => {
      const c = g.nodes[id].c;
      const col = kind === 'bwd' ? 'accent-2' : 'accent';
      c.style.fill = kind ? tint(col, 28) : 'var(--surface)';
      c.style.stroke = kind ? 'var(--' + col + ')' : 'var(--ink)';
      c.setAttribute('stroke-width', kind ? 3 : 1.5);
    };

    g.setAdj = (id, v) => {
      g.nodes[id].adj.textContent = h.fmt(v, 2);
      g.nodes[id].adj.style.opacity = 1;
    };

    g.showAll = (withAdj) => {
      NODES.forEach((n) => {
        g.nodes[n.id].val.style.opacity = 1;
        if (withAdj) g.setAdj(n.id, g.data.adj[n.id]);
        else g.nodes[n.id].adj.style.opacity = 0;
      });
      g.edges.forEach((E) => { E.lab.style.opacity = withAdj ? 1 : 0; E.hl.style.opacity = 0; });
      NODES.forEach((n) => g.mark(n.id, null));
    };

    g.hideAdj = () => {
      NODES.forEach((n) => { g.nodes[n.id].adj.style.opacity = 0; });
      g.edges.forEach((E) => { E.lab.style.opacity = 0; });
    };

    g.hideVals = () => {
      NODES.forEach((n) => { g.nodes[n.id].val.style.opacity = 0; });
    };

    g.forward = async (ok) => {
      for (const n of NODES) {
        if (!ok()) return;
        const inc = g.edges.filter((E) => E.e.b === n.id);
        inc.forEach((E) => { E.hl.style.stroke = 'var(--accent)'; E.hl.style.opacity = 1; });
        g.mark(n.id, 'fwd');
        await h.sleep(500);
        if (!ok()) return;
        g.vis(g.nodes[n.id].val, true);
        await h.sleep(550);
        inc.forEach((E) => { E.hl.style.opacity = 0; });
        g.mark(n.id, null);
      }
    };

    g.backward = async (ok, info) => {
      const acc = {};
      NODES.forEach((n) => { acc[n.id] = 0; });
      acc.sub = 1;
      g.setAdj('sub', 1);
      if (info) info('Graine : $\\partial f/\\partial f = 1$ sur le nœud de sortie.');
      await h.sleep(700);
      for (let i = NODES.length - 1; i >= 0; i--) {
        const n = NODES[i];
        if (!ok()) return;
        const inc = [];
        g.edges.forEach((E, k) => { if (E.e.b === n.id) inc.push({ E, k }); });
        g.mark(n.id, 'bwd');
        await h.sleep(450);
        for (const it of inc) {
          if (!ok()) return;
          const E = it.E;
          const c = g.data.contrib[it.k];
          E.hl.style.stroke = 'var(--accent-2)';
          E.hl.style.opacity = 1;
          E.lab.style.opacity = 1;
          const dot = h.el('circle', { cx: E.ex, cy: E.ey, r: 6 }, fxLayer);
          dot.style.fill = 'var(--accent-2)';
          await h.tween(650, (p) => h.attr(dot, { cx: h.lerp(E.ex, E.sx, p), cy: h.lerp(E.ey, E.sy, p) }));
          if (!ok()) return;
          dot.remove();
          acc[E.e.a] += c;
          g.setAdj(E.e.a, acc[E.e.a]);
          if (info) {
            info(NAMES[n.id] + ' → ' + NAMES[E.e.a] + ' : ' + h.fmt(acc[n.id], 2) + ' × ' + h.fmt(g.data.loc[it.k], 2) + ' = ' + h.fmt(c, 2) +
              (acc[E.e.a] !== c ? ' ; cumul sur ' + NAMES[E.e.a] + ' = ' + h.fmt(acc[E.e.a], 2) : ''));
          }
          await h.sleep(350);
          E.hl.style.opacity = 0;
        }
        g.mark(n.id, null);
      }
      if (info) info('Terminé : un seul parcours arrière a donné les deux dérivées.');
    };

    g.update(x1, x2);
    return g;
  }

  /**
   * Draws a rounded information panel with a fixed number of text lines; returns a setter.
   */
  function makePanel(h, svg, x, y, w, hh, n) {
    const r = h.el('rect', { x, y, width: w, height: hh, rx: 10, 'stroke-width': 1 }, svg);
    r.style.fill = 'var(--surface-2)';
    r.style.stroke = 'var(--line)';
    const lines = [];
    for (let i = 0; i < n; i++) {
      lines.push(T(h, svg, x + 12, y + 22 + i * ((hh - 30) / Math.max(1, n - 1)), '', { anchor: 'start', mono: true, size: 12.5, op: 1 }));
    }
    return (i, text, fill, size) => {
      lines[i].textContent = text;
      lines[i].style.fill = fill || 'var(--ink)';
      if (size) lines[i].style.fontSize = size + 'px';
    };
  }

  /**
   * Formats an integer with French thousand separators.
   */
  function frInt(x) {
    return Math.round(x).toLocaleString('fr-FR');
  }

  /**
   * Formats a duration in seconds into a short French string.
   */
  function frDuration(s) {
    const f = (x, d) => x.toFixed(d).replace('.', ',');
    if (s < 1) return f(s * 1000, 0) + ' ms';
    if (s < 60) return f(s, 1) + ' s';
    if (s < 3600) return f(s / 60, 1) + ' min';
    if (s < 86400) return f(s / 3600, 1) + ' h';
    if (s < 86400 * 365) return f(s / 86400, 1) + ' jours';
    return f(s / (86400 * 365), 1) + ' ans';
  }

  /**
   * Removes everything drawn in the stage so another view can be built.
   */
  function clearStage(h) {
    Array.from(h.stage.children).forEach((c) => c.remove());
  }

  /**
   * Returns the pair (symbolic derivative tree size, original tree size) after n compositions of h -> h sin h.
   */
  function swell(n) {
    let S = 1;
    let D = 1;
    for (let i = 0; i < n; i++) {
      const nd = 6 + 2 * D + 3 * S;
      S = 2 * S + 2;
      D = nd;
    }
    return { S, D, A: 3 * (2 * n + 1) };
  }

  Course.register({
    id: 'autodiff',
    order: 10,
    title: 'Implémentation : différentiation automatique',
    summary: 'Comment les bibliothèques (TensorFlow, Keras, PyTorch) calculent les gradients : différences finies, dérivation symbolique, puis <strong>différentiation automatique</strong> sur un graphe de calcul.',
    pdfPages: '34-37',
    steps: [
      {
        title: 'Trois façons de dériver',
        duration: 11000,
        text: String.raw`<p>La descente de gradient a besoin de $\nabla f$ à <strong>chaque</strong> itération, pour des millions de paramètres. Heureusement, les <em>toolboxes</em> calculent les gradients à notre place.</p>
<p>Il existe trois grandes approches. Comparons-les sur la même fonction :</p>
$$f(x_1,x_2)=\ln x_1 + x_1 x_2 - \sin x_2$$
<p>dont le gradient exact est $\nabla f = \big(\tfrac{1}{x_1}+x_2,\; x_1-\cos x_2\big)$.</p>`,
        note: 'Numérique : approximatif et coûteux. Symbolique : exact mais l\'expression gonfle. Automatique : exact et peu coûteux, c\'est ce qu\'utilisent les bibliothèques.',
        draw(h) {
          const svg = h.svg(720, 420);
          const top = h.label(svg, 40, 8, 640, 56, 'Objectif : calculer $\\nabla f(x_1,x_2)$ pour $f=\\ln x_1 + x_1x_2-\\sin x_2$', 'center');
          top.style.fontSize = '17px';
          const cards = [
            {
              title: 'Numérique', formula: '\\dfrac{f(x+h)-f(x)}{h}',
              props: [['bad', 'Approximatif : erreur de troncature et d\'arrondi'], ['bad', 'Coût : $n+1$ évaluations de $f$'], ['ok', 'Très simple à coder']]
            },
            {
              title: 'Symbolique', formula: '(uv)\'=u\'v+uv\'',
              props: [['ok', 'Exact'], ['bad', 'L\'expression de la dérivée <em>gonfle</em>'], ['warn', 'Peu adapté aux boucles et aux tests']]
            },
            {
              title: 'Automatique', formula: '\\dfrac{\\partial f}{\\partial x}=\\dfrac{\\partial f}{\\partial v}\\dfrac{\\partial v}{\\partial x}',
              props: [['ok', 'Exact (à l\'arrondi machine près)'], ['ok', 'Coût ≈ 2 à 3 évaluations de $f$'], ['ok', 'Utilisé par TensorFlow et PyTorch']]
            }
          ];
          const groups = cards.map((c, i) => {
            const gx = 20 + i * 235;
            const grp = h.el('g', {}, svg);
            grp.style.opacity = 0;
            grp.style.transform = 'translateY(16px)';
            grp.style.transition = 'opacity .6s, transform .6s';
            const r = h.el('rect', { x: gx, y: 85, width: 210, height: 315, rx: 12, 'stroke-width': 1.5 }, grp);
            r.style.fill = 'var(--surface-2)';
            r.style.stroke = 'var(--line)';
            r.style.transition = 'stroke .5s';
            T(h, grp, gx + 105, 115, c.title, { size: 16, weight: 700 });
            h.label(grp, gx + 8, 128, 194, 56, '$' + c.formula + '$', 'center');
            c.props.forEach((p, k) => {
              const col = p[0] === 'ok' ? 'var(--ok)' : p[0] === 'bad' ? 'var(--bad)' : 'var(--warn)';
              const mark = p[0] === 'ok' ? '✓' : p[0] === 'bad' ? '✗' : '!';
              h.label(grp, gx + 10, 195 + k * 66, 192, 62, '<span style="color:' + col + ';font-weight:700">' + mark + '</span> ' + p[1]);
            });
            return { grp, r };
          });
          (async () => {
            for (let i = 0; i < groups.length; i++) {
              await h.sleep(500);
              groups[i].grp.style.opacity = 1;
              groups[i].grp.style.transform = 'translateY(0)';
            }
            await h.sleep(900);
            groups[2].r.style.stroke = 'var(--accent)';
            groups[2].r.setAttribute('stroke-width', 3);
          })();
        }
      },
      {
        title: 'Différences finies',
        duration: 12000,
        text: String.raw`<p>L'idée la plus directe : approcher la dérivée par un taux d'accroissement,</p>
$$\frac{\partial f}{\partial x_1}\approx\frac{f(x_1+h,x_2)-f(x_1,x_2)}{h}.$$
<p>Si $h$ est trop grand, l'approximation est mauvaise (erreur de <strong>troncature</strong> $\propto h$). Si $h$ est trop petit, l'ordinateur soustrait deux nombres presque égaux : l'erreur d'<strong>arrondi</strong> explose ($\propto 1/h$). Il existe un compromis, vers $h\approx 10^{-8}$.</p>
<p>Et il faut <em>une évaluation de $f$ par paramètre</em> : $n+1$ évaluations pour un gradient. Déplacez le curseur.</p>`,
        note: 'Différences finies : erreur minimale vers $h\\approx10^{-8}$ (double précision) et coût $n+1$ évaluations. Utile surtout pour vérifier un gradient.',
        check: {
          q: 'Pourquoi ne peut-on pas prendre $h$ aussi petit que possible ?',
          choices: ['La formule n\'est exacte que pour $h$ grand', 'Les erreurs d\'arrondi (en $1/h$) finissent par dominer', 'La fonction n\'est plus dérivable', 'Le coût de calcul augmente quand $h$ diminue'],
          answer: 1,
          explain: 'On soustrait deux valeurs presque identiques : les chiffres significatifs partent, et l\'erreur d\'arrondi croît comme $\\varepsilon/h$.'
        },
        draw(h) {
          const a = 2;
          const b = 1;
          const exact = GRAD(a, b)[0];
          const X0 = 85;
          const X1 = 690;
          const Y0 = 40;
          const Y1 = 320;
          const sx = h.scale(-16, 0, X0, X1);
          const sy = h.scale(1, -17, Y0, Y1);
          const errAt = (p) => {
            const hh = Math.pow(10, p);
            const approx = (F(a + hh, b) - F(a, b)) / hh;
            return { approx, err: Math.abs(approx - exact) };
          };
          const ly = (e) => Math.log10(Math.max(e, 1e-17));
          const svg = h.svg(720, 420);
          [-16, -12, -8, -4, 0].forEach((p) => {
            h.el('line', { x1: sx(p), y1: Y0, x2: sx(p), y2: Y1, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
            h.label(svg, sx(p) - 26, Y1 + 4, 52, 22, '$10^{' + p + '}$', 'center');
          });
          [0, -5, -10, -15].forEach((t) => {
            h.el('line', { x1: X0, y1: sy(t), x2: X1, y2: sy(t), stroke: 'var(--line)', 'stroke-width': 1 }, svg);
            h.label(svg, 22, sy(t) - 11, 58, 22, '$10^{' + t + '}$', 'center');
          });
          h.el('line', { x1: X0, y1: Y1, x2: X1, y2: Y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          h.el('line', { x1: X0, y1: Y0, x2: X0, y2: Y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          h.label(svg, X0, Y1 + 28, X1 - X0, 24, 'pas $h$ (échelle logarithmique)', 'center');
          h.label(svg, 90, 6, 300, 26, 'erreur $|\\hat f\'-f\'|$');

          const trunc = [];
          const round = [];
          for (let p = -16; p <= 0.001; p += 0.5) {
            trunc.push([sx(p), sy(Math.log10(0.125) + p)]);
            round.push([sx(p), sy(Math.log10(3.2e-16) - p)]);
          }
          const toPath = (pts) => pts.map((q, i) => (i ? 'L' : 'M') + q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ');
          h.el('path', { d: toPath(trunc), fill: 'none', stroke: 'var(--warn)', 'stroke-width': 2, 'stroke-dasharray': '6 5' }, svg);
          h.el('path', { d: toPath(round), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2, 'stroke-dasharray': '6 5' }, svg);
          T(h, svg, sx(-3.2), sy(Math.log10(0.125) - 3.2) + 22, 'troncature ∝ h', { fill: 'var(--warn)', weight: 700, halo: true });
          T(h, svg, sx(-13.6), sy(Math.log10(3.2e-16) + 13.6) - 12, 'arrondi ∝ 1/h', { fill: 'var(--accent-2)', weight: 700, halo: true });

          const curve = h.el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.8, 'stroke-linejoin': 'round' }, svg);
          const pts = [];
          for (let p = -16; p <= 0.001; p += 0.1) pts.push([sx(p), sy(ly(errAt(p).err))]);
          const full = toPath(pts);
          const vline = h.el('line', { y1: Y0, y2: Y1, stroke: 'var(--ink)', 'stroke-width': 1.2, 'stroke-dasharray': '3 4' }, svg);
          const dot = h.el('circle', { r: 7, 'stroke-width': 2 }, svg);
          dot.style.fill = 'var(--accent)';
          dot.style.stroke = 'var(--surface)';
          const costLabel = h.label(svg, 40, 384, 640, 30, 'Coût : <strong>1 évaluation de $f$ par paramètre</strong>, donc $n+1$ évaluations pour $\\nabla f$.', 'center');
          costLabel.style.fontSize = '15px';

          const rd = h.readout('');
          const show = (p) => {
            const r = errAt(p);
            h.attr(vline, { x1: sx(p), x2: sx(p) });
            h.attr(dot, { cx: sx(p), cy: sy(ly(r.err)) });
            rd('$h=10^{' + h.fmt(p, 1) + '}$ · estimation : ' + h.fmt(r.approx, 6) + ' · exact : ' + h.fmt(exact, 6) + ' · erreur : ' + r.err.toExponential(1).replace('.', ','));
          };
          const sl = h.slider({ label: '$\\log_{10} h$', min: -16, max: 0, step: 0.1, value: -4, format: (v) => h.fmt(v, 1), onInput: show });
          show(-4);
          let drawn = false;
          h.tween(1800, (p) => {
            const n = Math.max(2, Math.round(p * pts.length));
            h.attr(curve, { d: toPath(pts.slice(0, n)) });
            if (p >= 1 && !drawn) { drawn = true; h.attr(curve, { d: full }); }
          });
          h.button('Aller à l\'optimum', () => { sl.set(-8); show(-8); });
        }
      },
      {
        title: 'Dérivation symbolique',
        duration: 14000,
        text: String.raw`<p>Un logiciel de calcul formel réécrit l'<strong>expression</strong> de $f$ en appliquant les règles de dérivation : somme, produit $(uv)'=u'v+uv'$, chaîne, dérivées usuelles. Le résultat est exact.</p>
<p>Le revers : sans partage des sous-expressions, la dérivée peut devenir <strong>beaucoup plus grosse</strong> que $f$. C'est l'<em>explosion de l'expression</em> (<em>expression swell</em>). Prenons $h_{k+1}=h_k\sin(h_k)$ composée $n$ fois : l'arbre de la dérivée explose, alors qu'un graphe de calcul garde une taille linéaire.</p>`,
        note: 'Symbolique : exact, mais l\'expression de la dérivée peut croître exponentiellement avec la profondeur du calcul.',
        check: {
          q: 'Qu\'appelle-t-on « explosion de l\'expression » en dérivation symbolique ?',
          choices: ['L\'erreur d\'arrondi devient infinie', 'La dérivée n\'existe plus', 'L\'expression de la dérivée est bien plus grosse que celle de $f$ (sous-expressions dupliquées)', 'Il faut plus de paramètres'],
          answer: 2,
          explain: 'La règle du produit et la règle de la chaîne dupliquent les sous-expressions, d\'où une croissance rapide de la taille de l\'expression.'
        },
        draw(h) {
          let mode = 'rules';
          let wvar = 1;
          let vt = 0;
          const rd = h.readout('');

          const tg = h.toggle([['rules', 'Règles de dérivation'], ['swell', 'Explosion']], 'rules', (m) => { mode = m; render(); });
          const vtg = h.toggle([[1, '$\\partial/\\partial x_1$'], [2, '$\\partial/\\partial x_2$']], 1, (v) => { wvar = +v; render(); });
          h.typeset(vtg);
          const sl = h.slider({ label: 'profondeur $n$', min: 1, max: 10, step: 1, value: 4, decimals: 0, onInput: () => { if (mode === 'swell') updateSwell(); } });
          const slWrap = sl.input.parentNode;
          let updateSwell = () => {};

          /**
           * Rebuilds the stage for the currently selected view.
           */
          function render() {
            vt++;
            const tok = vt;
            const ok = () => tok === vt && h.alive();
            clearStage(h);
            vtg.style.display = mode === 'rules' ? '' : 'none';
            slWrap.style.display = mode === 'swell' ? '' : 'none';
            rd('');
            if (mode === 'rules') rulesView(ok);
            else swellView(ok);
          }

          /**
           * Animated step-by-step symbolic differentiation of f.
           */
          function rulesView(ok) {
            const svg = h.svg(720, 420);
            const k = wvar;
            const d = '\\dfrac{\\partial}{\\partial x_' + k + '}';
            const blocks = [
              { tex: d + '\\Big(\\ln x_1 + x_1x_2 - \\sin x_2\\Big)', tag: 'on part de l\'expression de $f$' },
              { tex: '= \\dfrac{\\partial \\ln x_1}{\\partial x_' + k + '} + \\dfrac{\\partial (x_1x_2)}{\\partial x_' + k + '} - \\dfrac{\\partial \\sin x_2}{\\partial x_' + k + '}', tag: 'règle de la somme (linéarité)' },
              {
                tex: k === 1 ? '= \\dfrac{1}{x_1} + \\big(1\\cdot x_2 + x_1\\cdot 0\\big) - 0' : '= 0 + \\big(0\\cdot x_2 + x_1\\cdot 1\\big) - \\cos x_2',
                tag: 'dérivées usuelles et règle du produit $(uv)\'=u\'v+uv\'$'
              },
              { tex: k === 1 ? '= \\dfrac{1}{x_1} + x_2' : '= x_1 - \\cos x_2', tag: 'simplification : une nouvelle expression, évaluée ensuite en chaque point' }
            ];
            const els = blocks.map((b, i) => {
              const y = 20 + i * 92;
              const grp = h.el('g', {}, svg);
              grp.style.opacity = 0;
              grp.style.transition = 'opacity .6s';
              const lab = h.label(grp, 20, y, 680, 52, '$$' + b.tex + '$$', 'center');
              lab.style.fontSize = '16px';
              const tag = h.label(grp, 40, y + 54, 640, 24, b.tag, 'center');
              tag.style.color = 'var(--muted)';
              tag.style.fontSize = '13px';
              return grp;
            });
            (async () => {
              for (let i = 0; i < els.length; i++) {
                await h.sleep(i === 0 ? 300 : 1600);
                if (!ok()) return;
                els[i].style.opacity = 1;
              }
            })();
          }

          /**
           * Bar chart (log scale) of derivative expression size versus automatic differentiation.
           */
          function swellView(ok) {
            const svg = h.svg(720, 420);
            const X0 = 85;
            const X1 = 700;
            const Y0 = 30;
            const Y1 = 330;
            const maxD = swell(10).D;
            const K = Math.ceil(Math.log10(maxD));
            const sy = (v) => Y1 - (Math.log10(Math.max(1, v)) / K) * (Y1 - Y0);
            const slot = (X1 - X0) / 10;
            for (let t = 0; t <= K; t++) {
              h.el('line', { x1: X0, y1: sy(Math.pow(10, t)), x2: X1, y2: sy(Math.pow(10, t)), stroke: 'var(--line)', 'stroke-width': 1 }, svg);
              h.label(svg, 22, sy(Math.pow(10, t)) - 11, 58, 22, '$10^{' + t + '}$', 'center');
            }
            h.el('line', { x1: X0, y1: Y1, x2: X1, y2: Y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            h.label(svg, 90, 4, 300, 24, 'nombre de nœuds (échelle log)');
            h.label(svg, X0, Y1 + 24, X1 - X0, 22, 'profondeur $n$', 'center');
            const bars = [];
            for (let n = 1; n <= 10; n++) {
              const s = swell(n);
              const bx = X0 + (n - 1) * slot + 6;
              T(h, svg, bx + slot / 2 - 6, Y1 + 18, String(n), { size: 12, fill: 'var(--muted)' });
              const r1 = h.el('rect', { x: bx, y: Y1, width: 22, height: 0, rx: 3 }, svg);
              r1.style.fill = 'var(--bad)';
              r1.style.transition = 'opacity .3s';
              const r2 = h.el('rect', { x: bx + 24, y: Y1, width: 22, height: 0, rx: 3 }, svg);
              r2.style.fill = 'var(--ok)';
              r2.style.transition = 'opacity .3s';
              bars.push({ n, s, r1, r2, y1: sy(s.D), y2: sy(s.A) });
            }
            T(h, svg, 330, 372, '■', { fill: 'var(--bad)', size: 14 });
            T(h, svg, 345, 372, 'dérivée symbolique (arbre)', { anchor: 'start', size: 13 });
            T(h, svg, 330, 396, '■', { fill: 'var(--ok)', size: 14 });
            T(h, svg, 345, 396, 'différentiation automatique (graphe partagé, ordre de grandeur)', { anchor: 'start', size: 13 });
            let grown = 0;
            updateSwell = () => {
              const n = sl.get();
              bars.forEach((b) => {
                const on = b.n <= n;
                b.r1.style.opacity = on ? 1 : 0.2;
                b.r2.style.opacity = on ? 1 : 0.2;
              });
              const s = swell(n);
              rd('$n=' + n + '$ · arbre de $h_n$ : ' + frInt(s.S) + ' nœuds · dérivée symbolique : <strong>' + frInt(s.D) + '</strong> nœuds · autodiff : ≈ ' + frInt(s.A));
            };
            updateSwell();
            (async () => {
              await h.tween(1600, (p) => {
                if (!ok()) return;
                bars.forEach((b) => {
                  const hh1 = (Y1 - b.y1) * p;
                  const hh2 = (Y1 - b.y2) * p;
                  h.attr(b.r1, { y: Y1 - hh1, height: hh1 });
                  h.attr(b.r2, { y: Y1 - hh2, height: hh2 });
                });
              });
              grown = 1;
            })();
          }

          render();
        }
      },
      {
        title: 'Le graphe de calcul',
        duration: 12000,
        text: String.raw`<p>La différentiation automatique part d'une observation : tout programme est une suite d'<strong>opérations élémentaires</strong> dont on connaît la dérivée. On les organise en <strong>graphe de calcul</strong> :</p>
$$v_1=x_1,\; v_2=x_2,\; v_3=\ln v_1,\; v_4=v_1v_2,\; v_5=\sin v_2,\; v_6=v_3+v_4,\; v_7=v_6-v_5=f.$$
<p>Chaque flèche porte une <strong>dérivée locale</strong> $\partial(\text{enfant})/\partial(\text{parent})$ : simple, car elle ne concerne qu'une seule opération. Le gradient s'obtiendra en les combinant par la règle de la chaîne.</p>`,
        note: 'On ne dérive jamais l\'expression entière : seulement des opérations élémentaires, dont la dérivée locale est triviale.',
        draw(h) {
          const svg = h.svg(720, 420);
          const g = makeGraph(h, svg, 1.5, 1.0, {});
          g.setMode('sym');
          NODES.forEach((n) => g.vis(g.nodes[n.id].grp, false));
          g.edges.forEach((E) => g.vis(E.grp, false));
          g.vis(g.out, false);
          T(h, svg, 705, 396, 'étiquettes : dérivée locale ∂(enfant)/∂(parent)', { anchor: 'end', size: 13, fill: 'var(--muted)' });
          (async () => {
            for (const n of NODES) {
              await h.sleep(600);
              g.vis(g.nodes[n.id].grp, true);
              g.edges.forEach((E) => { if (E.e.b === n.id) g.vis(E.grp, true); });
            }
            await h.sleep(500);
            g.vis(g.out, true);
            await h.sleep(700);
            for (const E of g.edges) {
              E.lab.style.opacity = 1;
              E.hl.style.stroke = 'var(--accent-2)';
              E.hl.style.opacity = 1;
              await h.sleep(450);
              E.hl.style.opacity = 0;
            }
          })();
        }
      },
      {
        title: 'Passe avant',
        duration: 12000,
        text: String.raw`<p>On commence par la <strong>passe avant</strong> (<em>forward</em>) : on évalue les nœuds dans l'ordre, des entrées vers la sortie. Chaque nœud calcule sa valeur à partir de celles de ses parents.</p>
<p>Ces valeurs intermédiaires sont <em>mémorisées</em> : on en aura besoin pour les dérivées locales. Changez $x_1$ et $x_2$ : tout est recalculé.</p>`,
        note: 'La passe avant calcule $f$ et stocke toutes les valeurs intermédiaires pour la passe arrière.',
        draw(h) {
          const svg = h.svg(720, 420);
          const g = makeGraph(h, svg, 1.5, 1.0, { legend: 'val' });
          g.setMode('sym');
          g.hideVals();
          const set = makePanel(h, svg, 436, 322, 276, 86, 3);
          let run = 0;
          const sx1 = h.slider({ label: '$x_1$', min: 0.5, max: 3, step: 0.05, value: 1.5, onInput: () => change() });
          const sx2 = h.slider({ label: '$x_2$', min: 0, max: 3, step: 0.05, value: 1.0, onInput: () => change() });

          /**
           * Refreshes the result panel for the current inputs.
           */
          const panel = (done) => {
            const a = sx1.get();
            const b = sx2.get();
            set(0, 'x₁=' + h.fmt(a, 2) + '  x₂=' + h.fmt(b, 2));
            set(1, 'ln x₁ + x₁x₂ − sin x₂');
            set(2, done ? 'f = ' + h.fmt(F(a, b), 3) : 'f = …', 'var(--accent)', 17);
          };
          const change = () => {
            run++;
            g.update(sx1.get(), sx2.get());
            g.showAll(false);
            panel(true);
          };
          panel(false);
          const play = () => {
            run++;
            const tok = run;
            g.update(sx1.get(), sx2.get());
            g.hideVals();
            panel(false);
            h.after(500, async () => {
              await g.forward(() => tok === run && h.alive());
              if (tok === run) panel(true);
            });
          };
          h.button('Rejouer la passe avant', play, 'primary');
          play();
        }
      },
      {
        title: 'Passe arrière',
        duration: 20000,
        text: String.raw`<p>La <strong>passe arrière</strong> (<em>reverse mode</em>, c'est la rétropropagation) part de la sortie avec la graine $\partial f/\partial f=1$. Pour chaque nœud, on note $\bar v=\partial f/\partial v$ (son <em>adjoint</em>) et on propage vers les parents :</p>
$$\bar v_{\text{parent}} \mathrel{+}= \bar v_{\text{enfant}}\times\frac{\partial v_{\text{enfant}}}{\partial v_{\text{parent}}}.$$
<p>Quand un nœud a <strong>plusieurs enfants</strong> ($x_1$ et $x_2$ ici), les contributions de tous les chemins s'<strong>additionnent</strong>. À la fin, on lit le gradient sur les entrées et on le compare au calcul analytique.</p>`,
        note: 'Un seul parcours arrière donne toutes les dérivées partielles : $\\bar x_1=\\frac1{x_1}+x_2$ et $\\bar x_2=x_1-\\cos x_2$, sommes des chemins.',
        check: {
          q: 'Le nœud $x_1$ a deux enfants ($\\ln$ et $\\times$). Que fait la rétropropagation de leurs contributions ?',
          choices: ['Elle garde la plus grande', 'Elle en fait la moyenne', 'Elle les additionne (règle de la chaîne sur tous les chemins)', 'Elle ignore l\'une des deux'],
          answer: 2,
          explain: 'La dérivée totale suit tous les chemins de $x_1$ vers $f$ : on additionne la contribution de chacun.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const g = makeGraph(h, svg, 1.5, 1.0, { legend: 'both' });
          g.setMode('num');
          g.showAll(false);
          g.hideAdj();
          const set = makePanel(h, svg, 436, 318, 276, 92, 3);
          const info = h.readout('');
          let run = 0;
          const sx1 = h.slider({ label: '$x_1$', min: 0.5, max: 3, step: 0.05, value: 1.5, onInput: () => change() });
          const sx2 = h.slider({ label: '$x_2$', min: 0, max: 3, step: 0.05, value: 1.0, onInput: () => change() });

          /**
           * Fills the comparison panel between autodiff and analytic gradients.
           */
          const panel = (done) => {
            const a = sx1.get();
            const b = sx2.get();
            const an = GRAD(a, b);
            if (!done) {
              set(0, 'autodiff : …');
              set(1, 'analytique : (' + h.fmt(an[0], 4) + ' ; ' + h.fmt(an[1], 4) + ')', 'var(--muted)');
              set(2, '');
              return;
            }
            const ad = g.data.adj;
            const diff = Math.max(Math.abs(ad.x1 - an[0]), Math.abs(ad.x2 - an[1]));
            set(0, 'autodiff : (' + h.fmt(ad.x1, 4) + ' ; ' + h.fmt(ad.x2, 4) + ')', 'var(--accent-2)');
            set(1, 'analytique : (' + h.fmt(an[0], 4) + ' ; ' + h.fmt(an[1], 4) + ')');
            set(2, diff < 1e-9 ? '✓ identiques' : 'écart ' + diff.toExponential(1), 'var(--ok)', 14);
          };
          const change = () => {
            run++;
            g.update(sx1.get(), sx2.get());
            g.showAll(true);
            panel(true);
            info('Gradient recalculé pour $x_1=' + h.fmt(sx1.get(), 2) + ',\\ x_2=' + h.fmt(sx2.get(), 2) + '$.');
          };
          const play = () => {
            run++;
            const tok = run;
            g.update(sx1.get(), sx2.get());
            g.hideAdj();
            g.showAll(false);
            panel(false);
            h.after(600, async () => {
              await g.backward(() => tok === run && h.alive(), (m) => { if (tok === run) info(m); });
              if (tok === run) panel(true);
            });
          };
          h.button('Rejouer la passe arrière', play, 'primary');
          play();
        }
      },
      {
        title: 'Pourquoi le mode inverse ?',
        duration: 14000,
        text: String.raw`<p>En apprentissage, la perte est un <strong>scalaire</strong> mais les paramètres se comptent en millions. Comparons le coût du gradient complet :</p>
<ul>
<li><strong>différences finies</strong> : une évaluation de $f$ par paramètre, soit $n+1$ évaluations ;</li>
<li><strong>mode inverse</strong> : une passe avant puis une passe arrière, dont le coût est du même ordre que $f$, soit environ $3$ évaluations, <em>quel que soit $n$</em>.</li>
</ul>
<p>C'est ce qui rend l'entraînement des grands réseaux possible. Faites varier $n$ ou cliquez sur un exemple.</p>`,
        note: 'Pour une perte scalaire, le mode inverse donne le gradient par rapport à <em>tous</em> les paramètres pour un coût constant (≈ 2 à 3 évaluations de $f$).',
        check: {
          q: 'Un réseau a $10^6$ paramètres. Combien d\'évaluations de $f$ (ordre de grandeur) pour le gradient par différences finies, puis en mode inverse ?',
          choices: ['$10^6+1$ puis ≈ 3', '≈ 3 puis $10^6$', '$10^6+1$ dans les deux cas', '2 dans les deux cas'],
          answer: 0,
          explain: 'Les différences finies coûtent $n+1$ évaluations ; le mode inverse reste à quelques passes, indépendamment de $n$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const X0 = 40;
          const BW = 520;
          T(h, svg, X0, 62, 'Différences finies : n + 1 évaluations de f', { anchor: 'start', size: 15, weight: 700 });
          T(h, svg, X0, 172, 'Mode inverse : 1 passe avant + 1 passe arrière ≈ 3 évaluations', { anchor: 'start', size: 15, weight: 700 });
          const r1 = h.el('rect', { x: X0, y: 74, width: BW, height: 42, rx: 6 }, svg);
          r1.style.fill = 'var(--bad)';
          const r2 = h.el('rect', { x: X0, y: 184, width: 4, height: 42, rx: 6 }, svg);
          r2.style.fill = 'var(--ok)';
          const v1 = T(h, svg, X0 + BW + 12, 101, '', { anchor: 'start', mono: true, size: 14, weight: 700 });
          const v2 = T(h, svg, X0 + 20, 211, '', { anchor: 'start', mono: true, size: 14, weight: 700 });
          h.el('line', { x1: 40, y1: 262, x2: 680, y2: 262, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
          T(h, svg, 360, 288, 'si une évaluation de f dure 10 ms :', { size: 13, fill: 'var(--muted)' });
          const t1 = T(h, svg, 190, 322, '', { size: 16, weight: 700, fill: 'var(--bad)' });
          const t2 = T(h, svg, 530, 322, '', { size: 16, weight: 700, fill: 'var(--ok)' });
          T(h, svg, 190, 342, 'différences finies', { size: 12, fill: 'var(--muted)' });
          T(h, svg, 530, 342, 'mode inverse', { size: 12, fill: 'var(--muted)' });
          const ratio = T(h, svg, 360, 390, '', { size: 17, weight: 700 });
          const rd = h.readout('');
          const show = (p) => {
            const n = Math.round(Math.pow(10, p));
            const c1 = n + 1;
            const c2 = 3;
            const m = Math.max(c1, c2);
            const w1 = Math.max(4, BW * (c1 / m));
            const w2 = Math.max(4, BW * (c2 / m));
            h.attr(r1, { width: w1 });
            h.attr(r2, { width: w2 });
            v1.textContent = frInt(c1);
            v2.textContent = '≈ 3';
            v2.setAttribute('x', X0 + w2 + 12);
            t1.textContent = frDuration(c1 * 0.01);
            t2.textContent = frDuration(c2 * 0.01);
            ratio.textContent = 'mode inverse : ≈ ' + frInt(c1 / c2) + ' × moins cher';
            rd('$n=' + frInt(n).replace(/ | /g, '\\,') + '$ paramètres');
          };
          const sl = h.slider({ label: '$\\log_{10} n$', min: 0, max: 11, step: 0.05, value: 6, format: (v) => h.fmt(v, 2), onInput: show });
          [['LeNet-5', 4.78], ['ResNet-50', 7.4], ['GPT-3', 11.24]].forEach(([name, p]) => {
            h.button(name, () => { sl.set(Math.min(11, p)); show(Math.min(11, p)); });
          });
          show(6);
        }
      },
      {
        title: 'Toolboxes et GPU',
        duration: 15000,
        text: String.raw`<p>En pratique, on ne programme pas la rétropropagation : les <strong>toolboxes</strong> Python le font. <strong>TensorFlow</strong> (Google), <strong>Keras</strong> (Google), une surcouche de haut niveau de TensorFlow (et de Theano), et <strong>PyTorch</strong> (Meta).</p>
<p>Les calculs tournent sur <strong>GPU</strong> grâce aux pilotes NVIDIA (<strong>CUDA</strong>, <strong>cuDNN</strong>) : des milliers de petits cœurs calculent un produit matriciel en parallèle.</p>
<p>Même petit MLP, avec la ligne où a lieu la différentiation automatique :</p>
<pre style="overflow-x:auto;background:var(--surface-2);padding:8px 10px;border-radius:8px;font-size:12px;line-height:1.45"><code>model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(1)])
with tf.GradientTape() as tape:
    loss = mse(y, model(x))
<b style="color:var(--accent-2)">grads = tape.gradient(loss, model.trainable_variables)</b></code></pre>
<pre style="overflow-x:auto;background:var(--surface-2);padding:8px 10px;border-radius:8px;font-size:12px;line-height:1.45"><code>model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
loss = ((model(x) - y) ** 2).mean()
<b style="color:var(--accent-2)">loss.backward()</b>
optimizer.step()</code></pre>`,
        note: 'Keras, TensorFlow et PyTorch implémentent le mode inverse (<code>GradientTape</code>, <code>loss.backward()</code>) et délèguent les calculs au GPU via CUDA et cuDNN.',
        check: {
          q: 'En PyTorch, quelle instruction déclenche la différentiation automatique (passe arrière) ?',
          choices: ['<code>optimizer.step()</code>', '<code>loss.backward()</code>', '<code>nn.Linear(2, 16)</code>', '<code>model(x)</code>'],
          answer: 1,
          explain: '<code>loss.backward()</code> parcourt le graphe de calcul en sens inverse et remplit les gradients ; <code>optimizer.step()</code> les utilise ensuite pour mettre à jour les poids.'
        },
        draw(h) {
          let mode = 'lib';
          let vt = 0;
          h.toggle([['lib', 'Bibliothèques'], ['gpu', 'CPU contre GPU']], 'lib', (m) => { mode = m; render(); });

          /**
           * Rebuilds the stage for the selected view.
           */
          function render() {
            vt++;
            const tok = vt;
            const ok = () => tok === vt && h.alive();
            clearStage(h);
            if (mode === 'lib') libView(ok);
            else gpuView(ok);
          }

          /**
           * Draws a rounded box with a title and an optional subtitle inside a group.
           */
          function box(parent, x, y, w, hh, title, sub, o) {
            const opt = o || {};
            const grp = h.el('g', {}, parent);
            grp.style.opacity = 0;
            grp.style.transition = 'opacity .6s';
            const r = h.el('rect', { x, y, width: w, height: hh, rx: 10, 'stroke-width': opt.stroke ? 2.5 : 1.5 }, grp);
            r.style.fill = opt.fill || 'var(--surface-2)';
            r.style.stroke = opt.stroke || 'var(--ink)';
            if (opt.dash) r.setAttribute('stroke-dasharray', '5 4');
            T(h, grp, x + w / 2, y + (sub ? hh / 2 - 2 : hh / 2 + 5), title, { size: 14, weight: 700 });
            if (sub) T(h, grp, x + w / 2, y + hh / 2 + 15, sub, { size: 12, fill: 'var(--muted)' });
            return grp;
          }

          /**
           * Layered view of the software stack from the model down to the GPU.
           */
          function libView(ok) {
            const svg = h.svg(720, 420);
            const layers = [
              [box(svg, 355, 355, 325, 48, 'GPU NVIDIA', 'milliers de cœurs en parallèle'), box(svg, 40, 355, 295, 48, 'CPU', 'quelques gros cœurs')],
              [box(svg, 40, 292, 640, 50, 'CUDA · cuDNN', 'pilotes et bibliothèques NVIDIA')],
              [box(svg, 40, 228, 640, 50, 'Différentiation automatique', 'graphe de calcul, passe avant et arrière', { stroke: 'var(--accent)', fill: tint('accent', 12) })],
              [box(svg, 40, 140, 190, 74, 'TensorFlow', 'Google'), box(svg, 240, 140, 95, 74, 'Theano', 'historique', { dash: true }), box(svg, 395, 90, 285, 124, 'PyTorch', 'Meta')],
              [box(svg, 40, 90, 295, 40, 'Keras', 'surcouche de TensorFlow et Theano'), box(svg, 40, 20, 640, 56, 'Votre modèle (Python)', 'couches, perte, optimiseur')]
            ];
            (async () => {
              for (const row of layers) {
                await h.sleep(700);
                if (!ok()) return;
                row.forEach((b) => { b.style.opacity = 1; });
              }
            })();
          }

          /**
           * Parallel matrix product animation: a few big cores versus many tiny ones.
           */
          function gpuView(ok) {
            const svg = h.svg(720, 420);
            const panels = [
              { x: 20, title: 'CPU : quelques gros cœurs', cores: 4, mx: 175 },
              { x: 370, title: 'GPU : des milliers de petits cœurs', cores: 48, mx: 525 }
            ];
            const N = 6;
            const cell = 26;
            const state = panels.map((p, pi) => {
              const r = h.el('rect', { x: p.x, y: 14, width: 330, height: 322, rx: 12, 'stroke-width': 1.5 }, svg);
              r.style.fill = 'var(--surface-2)';
              r.style.stroke = 'var(--line)';
              T(h, svg, p.x + 165, 42, p.title, { size: 14, weight: 700 });
              const cores = [];
              if (pi === 0) {
                for (let i = 0; i < 4; i++) {
                  const c = h.el('rect', { x: p.x + 22 + (i % 2) * 58, y: 90 + Math.floor(i / 2) * 58, width: 48, height: 48, rx: 8, 'stroke-width': 1.5 }, svg);
                  c.style.fill = 'var(--surface)';
                  c.style.stroke = 'var(--ink)';
                  c.style.transition = 'fill .2s';
                  cores.push(c);
                }
              } else {
                for (let i = 0; i < 48; i++) {
                  const c = h.el('rect', { x: p.x + 18 + (i % 8) * 17, y: 90 + Math.floor(i / 8) * 17, width: 13, height: 13, rx: 2, 'stroke-width': 1 }, svg);
                  c.style.fill = 'var(--surface)';
                  c.style.stroke = 'var(--ink)';
                  c.style.transition = 'fill .2s';
                  cores.push(c);
                }
              }
              T(h, svg, p.mx + (N * cell) / 2, 80, 'C = A·B', { size: 13, weight: 600 });
              const cells = [];
              for (let i = 0; i < N * N; i++) {
                const c = h.el('rect', { x: p.mx + (i % N) * cell, y: 90 + Math.floor(i / N) * cell, width: cell - 2, height: cell - 2, rx: 3, 'stroke-width': 1 }, svg);
                c.style.fill = 'var(--surface)';
                c.style.stroke = 'var(--line)';
                c.style.transition = 'fill .25s';
                cells.push(c);
              }
              const steps = T(h, svg, p.x + 165, 300, '', { size: 15, weight: 700, mono: true, fill: pi === 0 ? 'var(--bad)' : 'var(--ok)' });
              return { cores, cells, steps };
            });
            const cap = h.label(svg, 20, 346, 680, 60, 'Chaque case $C_{ij}=\\sum_k A_{ik}B_{kj}$ est indépendante des autres : on peut toutes les calculer <strong>en parallèle</strong>.', 'center');
            cap.style.fontSize = '15px';
            (async () => {
              while (ok()) {
                state.forEach((s) => {
                  s.cells.forEach((c) => { c.style.fill = 'var(--surface)'; });
                  s.steps.textContent = 'étapes : 0';
                });
                await h.sleep(500);
                for (let k = 0; k < 9; k++) {
                  if (!ok()) return;
                  const cpu = state[0];
                  const gpu = state[1];
                  cpu.cores.forEach((c) => { c.style.fill = tint('accent', 55); });
                  for (let j = 4 * k; j < 4 * k + 4; j++) cpu.cells[j].style.fill = tint('accent', 55);
                  cpu.steps.textContent = 'étapes : ' + (k + 1);
                  if (k === 0) {
                    gpu.cores.forEach((c) => { c.style.fill = tint('accent', 55); });
                    gpu.cells.forEach((c) => { c.style.fill = tint('accent', 55); });
                    gpu.steps.textContent = 'étapes : 1';
                  }
                  await h.sleep(260);
                  cpu.cores.forEach((c) => { c.style.fill = 'var(--surface)'; });
                  if (k === 0) gpu.cores.forEach((c) => { c.style.fill = 'var(--surface)'; });
                  await h.sleep(260);
                }
                await h.sleep(1800);
              }
            })();
          }

          render();
        }
      },
      {
        title: 'Les TPs',
        duration: 9000,
        text: String.raw`<p>Les séances de <strong>TP</strong> se font avec des <strong>notebooks Jupyter</strong>, en utilisant <strong>Keras / TensorFlow</strong>.</p>
<ul>
<li>On les fait <strong>ensemble pendant les séances</strong> ;</li>
<li>on les <strong>continue ou termine</strong> pour les séances suivantes.</li>
</ul>`,
        note: 'TPs : notebooks Jupyter, Keras / TensorFlow, faits en séance puis à finir pour la suite.',
        draw(h) {
          const svg = h.svg(720, 420);
          const card = h.el('rect', { x: 40, y: 12, width: 640, height: 300, rx: 12, 'stroke-width': 1.5 }, svg);
          card.style.fill = 'var(--surface)';
          card.style.stroke = 'var(--ink)';
          const head = h.el('rect', { x: 40, y: 12, width: 640, height: 30, rx: 12 }, svg);
          head.style.fill = 'var(--surface-2)';
          T(h, svg, 60, 32, 'TP_reseau.ipynb  ·  Jupyter', { anchor: 'start', size: 13, weight: 600 });
          const cells = [];
          const cellBox = (y, hh) => {
            const grp = h.el('g', {}, svg);
            grp.style.opacity = 0;
            grp.style.transition = 'opacity .6s';
            const r = h.el('rect', { x: 56, y, width: 608, height: hh, rx: 6, 'stroke-width': 1 }, grp);
            r.style.fill = 'var(--surface-2)';
            r.style.stroke = 'var(--line)';
            cells.push(grp);
            return grp;
          };
          const c1 = cellBox(52, 34);
          T(h, c1, 72, 74, 'TP 1 — Construire et entraîner un réseau avec Keras', { anchor: 'start', size: 14, weight: 700 });
          const c2 = cellBox(94, 92);
          T(h, c2, 70, 112, 'In [1]:', { anchor: 'start', mono: true, size: 12, fill: 'var(--accent)' });
          ['model = keras.Sequential([', '    layers.Dense(16, activation="relu"), layers.Dense(1)])', 'model.compile(optimizer="sgd", loss="mse")', 'model.fit(X, y, epochs=30)'].forEach((l, i) => {
            T(h, c2, 130, 112 + i * 17, l, { anchor: 'start', mono: true, size: 12 });
          });
          const c3 = cellBox(194, 108);
          T(h, c3, 70, 212, 'Out:', { anchor: 'start', mono: true, size: 12, fill: 'var(--accent)' });
          const cx0 = 130;
          const cx1 = 640;
          const cy0 = 214;
          const cy1 = 290;
          h.el('line', { x1: cx0, y1: cy1, x2: cx1, y2: cy1, stroke: 'var(--muted)', 'stroke-width': 1 }, c3);
          h.el('line', { x1: cx0, y1: cy0, x2: cx0, y2: cy1, stroke: 'var(--muted)', 'stroke-width': 1 }, c3);
          T(h, c3, cx1, cy1 + 0, '', {});
          T(h, c3, cx1 - 4, cy0 + 14, 'perte au fil des époques', { anchor: 'end', size: 12, fill: 'var(--muted)' });
          const curve = h.el('path', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, c3);
          const loss = (t) => 0.1 + 0.9 * Math.exp(-4 * t);
          const pts = [];
          for (let i = 0; i <= 60; i++) {
            const t = i / 60;
            pts.push([h.lerp(cx0, cx1, t), h.lerp(cy1 - 4, cy0 + 4, (loss(t) - 0.1) / 0.9 * 0.9 + 0.05)]);
          }
          const flow = [['Séance', 'on le fait ensemble'], ['À la maison', 'continuer / finir'], ['Séance suivante', 'on reprend']];
          const chips = flow.map((f, i) => {
            const x = 40 + i * 235;
            const grp = h.el('g', {}, svg);
            grp.style.opacity = 0;
            grp.style.transition = 'opacity .6s';
            const r = h.el('rect', { x, y: 336, width: 170, height: 60, rx: 12, 'stroke-width': 1.5 }, grp);
            r.style.fill = tint('accent', 14);
            r.style.stroke = 'var(--accent)';
            T(h, grp, x + 85, 361, f[0], { size: 14, weight: 700 });
            T(h, grp, x + 85, 381, f[1], { size: 12, fill: 'var(--muted)' });
            if (i < 2) h.el('line', { x1: x + 176, y1: 366, x2: x + 229, y2: 366, stroke: 'var(--ink)', 'stroke-width': 1.8, 'marker-end': 'url(#arrow)' }, grp);
            return grp;
          });
          (async () => {
            for (const c of cells) {
              await h.sleep(500);
              c.style.opacity = 1;
            }
            await h.tween(2000, (p) => {
              const n = Math.max(2, Math.round(p * pts.length));
              h.attr(curve, { d: pts.slice(0, n).map((q, i) => (i ? 'L' : 'M') + q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ') });
            });
            for (const c of chips) {
              await h.sleep(450);
              c.style.opacity = 1;
            }
          })();
        }
      }
    ]
  });
})();
