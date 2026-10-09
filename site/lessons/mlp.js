/**
 * Lesson: multilayer perceptron (layers, activations, vanishing gradient, non-linearity, XOR, architecture).
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
      txt(h, g, b.x0 - 9, y, o.yf ? o.yf(v) : tick(v), { size: 12, fill: 'var(--muted)', anchor: 'end', mid: true });
    });
    h.el('line', { x1: b.x0, y1: b.y1, x2: b.x1, y2: b.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    h.el('line', { x1: b.x0, y1: b.y0, x2: b.x0, y2: b.y1, stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    if (o.xl) txt(h, g, (b.x0 + b.x1) / 2, b.y1 + 38, o.xl, { size: 13, fill: 'var(--muted)' });
    if (o.yl) txt(h, g, b.x0 - 4, b.y0 - 10, o.yl, { size: 13, fill: 'var(--muted)', anchor: 'start' });
  }

  /**
   * Computes node positions of a fully connected network, one column per layer.
   */
  function layout(sizes, xl, xr, yt, yb, maxGap) {
    const mid = (yt + yb) / 2;
    return sizes.map((n, i) => {
      const x = sizes.length === 1 ? (xl + xr) / 2 : xl + (xr - xl) * i / (sizes.length - 1);
      const gap = n > 1 ? Math.min(maxGap || 56, (yb - yt) / (n - 1)) : 0;
      return Array.from({ length: n }, (_, j) => ({ x, y: mid + (j - (n - 1) / 2) * gap }));
    });
  }

  /**
   * Builds a matrix display (brackets plus text cells) translated to x,y; set(M) fills the cells.
   */
  function mat(h, parent, x, y, rows, cols, cw, ch, d, label) {
    const g = h.el('g', { transform: `translate(${x},${y})` }, parent);
    const w = cols * cw, hh = rows * ch;
    h.el('path', { d: `M6,0 H0 V${hh} H6`, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    h.el('path', { d: `M${w - 6},0 H${w} V${hh} H${w - 6}`, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
    const cells = [];
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) cells.push(txt(h, g, cw * (j + 0.5), ch * (i + 0.5), '', { size: 12, mono: true, mid: true }));
    }
    if (label) txt(h, g, w / 2, -9, label, { size: 12, fill: 'var(--muted)' });
    return {
      g, w, hgt: hh,
      set(M) {
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) cells[i * cols + j].textContent = h.fmt(M[i][j], d);
      },
      move(tx, ty) { g.setAttribute('transform', `translate(${tx},${ty})`); }
    };
  }

  /**
   * Turns a flat vector into a one-column matrix.
   */
  function col(v) { return v.map((a) => [a]); }

  /**
   * Clicks the i-th button of a segmented toggle (used to script its animation).
   */
  function pick(wrap, i) { wrap.children[i].click(); }

  const relu = (x) => Math.max(0, x);

  const FN = {
    sigmoid: {
      name: 'Sigmoïde',
      g: (x) => 1 / (1 + Math.exp(-x)),
      d: (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
      sat: true
    },
    tanh: {
      name: 'tanh',
      g: (x) => Math.tanh(x),
      d: (x) => 1 - Math.tanh(x) * Math.tanh(x),
      sat: true
    },
    relu: {
      name: 'ReLU',
      g: relu,
      d: (x) => (x > 0 ? 1 : 0),
      sat: false
    }
  };

  Course.register({
    id: 'mlp',
    order: 3,
    title: 'Perceptron multicouche (MLP)',
    summary: 'On empile des neurones en couches : <b>MLP</b>, non-linéarités dérivables (sigmoïde, tanh, ReLU), pourquoi elles sont indispensables (XOR) et comment choisir l\'architecture.',
    pdfPages: '16-18',
    steps: [
      {
        title: 'Du neurone au MLP',
        duration: 9000,
        text: String.raw`<p>Un seul neurone calcule $f = g(w^\top x)$ : c'est le <b>perceptron</b>.</p>
<p>Si on met <b>plusieurs neurones côte à côte</b> sur les mêmes entrées, on obtient une <b>couche</b> : chaque neurone a sa ligne de poids, regroupées dans une matrice $W$.</p>
<p>Si on <b>empile</b> les couches, la sortie d'une couche devient l'entrée de la suivante : c'est le <b>perceptron multicouche</b> (MLP).</p>`,
        note: String.raw`Un MLP est une composition de fonctions : $f(x) = g\big(W\,g(W_h\,x)\big)$. Chaque couche fait « produit matrice-vecteur puis non-linéarité ».`,
        draw(h) {
          const svg = h.svg(720, 420);
          const g = h.el('g', {}, svg);
          const defs = [[3, 1], [3, 2], [3, 4, 2]];
          const panels = [
            String.raw`<div style="font-size:15px"><b>Un neurone</b> (perceptron)</div>$$f = g(w^\top x)$$<p>$w \in \mathbb{R}^3$ : un vecteur de poids.</p>`,
            String.raw`<div style="font-size:15px"><b>Une couche</b> de neurones</div>$$f = g(W\,x)$$<p>$W \in \mathbb{R}^{2\times 3}$ : une ligne de poids par neurone.</p>`,
            String.raw`<div style="font-size:15px"><b>Plusieurs couches</b> (MLP)</div>$$f = g\big(W\,g(W_h\,x)\big)$$<p>$W_h$ : entrée vers cachée, $W$ : cachée vers sortie.</p>`
          ];
          let prev = null, cur = 0, seq = 0;

          const show = async (state) => {
            const sizes = defs[state];
            const pos = layout(sizes, 70, 400, 60, 360, 70);
            while (g.firstChild) g.removeChild(g.firstChild);
            const isNew = (li, j) => {
              if (prev === null) return false;
              if (prev.length !== sizes.length) return li > 0;
              return j >= prev[li];
            };
            const newEdges = [], newNodes = [];
            for (let li = 0; li < sizes.length - 1; li++) {
              pos[li].forEach((a, i) => pos[li + 1].forEach((b, j) => {
                const e = h.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'var(--muted)', 'stroke-width': 1.5 }, g);
                if (isNew(li, i) || isNew(li + 1, j) || (prev && prev.length !== sizes.length)) newEdges.push(e);
              }));
            }
            if (state === 0) {
              pos[0].forEach((a, i) => {
                const b = pos[1][0];
                txt(h, g, h.lerp(a.x, b.x, 0.3), h.lerp(a.y, b.y, 0.3) - 7, 'w' + '₁₂₃'[i], { size: 12, fill: 'var(--muted)' });
              });
            }
            pos.forEach((column, li) => column.forEach((p, j) => {
              const c = h.el('circle', { cx: p.x, cy: p.y, r: 20, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
              if (isNew(li, j)) newNodes.push(c);
              const label = li === 0 ? 'x' + '₁₂₃'[j] : 'g';
              txt(h, g, p.x, p.y, label, { size: 14, mid: true, fill: li === 0 ? 'var(--ink)' : 'var(--accent)', bold: li > 0 });
            }));
            const last = pos[pos.length - 1];
            last.forEach((p, j) => txt(h, g, p.x + 32, p.y, last.length === 1 ? 'f' : 'f' + '₁₂'[j], { size: 14, anchor: 'start', mid: true }));
            const lab = h.label(svg, 455, 90, 255, 250, panels[state]);
            g.appendChild(lab.parentNode);
            prev = sizes;
            cur = state;
            const id = ++seq;
            newNodes.forEach((c) => c.setAttribute('r', 0));
            newEdges.forEach((e) => e.setAttribute('opacity', 0));
            lab.style.opacity = 0;
            await h.tween(650, (p) => {
              if (id !== seq) return;
              newNodes.forEach((c) => c.setAttribute('r', 20 * p));
              newEdges.forEach((e) => e.setAttribute('opacity', p));
              lab.style.opacity = p;
            });
          };

          let run = 0;
          const wrap = h.toggle([[0, 'Un neurone'], [1, 'Une couche'], [2, 'Plusieurs couches']], 0, (v) => { show(v); });
          wrap.addEventListener('pointerdown', () => { run++; });
          h.button('▶ Faire grandir', async () => {
            const id = ++run;
            pick(wrap, 0);
            await h.sleep(1500);
            if (id !== run) return;
            pick(wrap, 1);
            await h.sleep(1800);
            if (id !== run) return;
            pick(wrap, 2);
          }, 'primary');
          show(0);
          const id0 = ++run;
          (async () => {
            await h.sleep(1800);
            if (id0 !== run) return;
            pick(wrap, 1);
            await h.sleep(2200);
            if (id0 !== run) return;
            pick(wrap, 2);
          })();
        }
      },
      {
        title: 'Passe avant chiffrée',
        duration: 11000,
        text: String.raw`<p>Calculer la sortie d'un MLP s'appelle la <b>passe avant</b> (<i>forward pass</i>). On avance couche par couche.</p>
<p>Pour chaque couche : un <b>produit matrice-vecteur</b> $z = W\,a$, puis la non-linéarité $a' = g(z)$ appliquée terme à terme.</p>
<p>Ici la couche cachée utilise ReLU et la sortie est linéaire. Bougez les curseurs : tout se recalcule.</p>`,
        note: String.raw`Toute la passe avant est une suite de produits matrice-vecteur : c'est pour cela que les GPU, excellents en algèbre linéaire, accélèrent tant l'apprentissage.`,
        draw(h) {
          const svg = h.svg(720, 420);
          const W1 = [[0.8, -0.4], [0.3, 0.9], [-0.6, 0.5]];
          const W2 = [[1, -1, 0.5], [0.4, 0.6, -0.8]];
          const X = [1, 0.5];
          const forward = () => {
            const z = W1.map((r) => r[0] * X[0] + r[1] * X[1]);
            const a = z.map(relu);
            const y = W2.map((r) => r[0] * a[0] + r[1] * a[1] + r[2] * a[2]);
            return { z, a, y };
          };
          const pos = layout([2, 3, 2], 60, 330, 80, 350, 80);
          const eg = [h.el('g', { style: 'transition:opacity .5s' }, svg), h.el('g', { style: 'transition:opacity .5s' }, svg)];
          const edgeLines = [[], []];
          [W1, W2].forEach((W, li) => {
            pos[li].forEach((a, i) => pos[li + 1].forEach((b, j) => {
              const w = W[j][i];
              edgeLines[li].push(h.el('line', {
                x1: a.x, y1: a.y, x2: b.x, y2: b.y,
                stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': 1 + 3 * Math.abs(w), opacity: 0.75
              }, eg[li]));
            }));
          });
          const ng = [h.el('g', {}, svg), h.el('g', { style: 'transition:opacity .5s' }, svg), h.el('g', { style: 'transition:opacity .5s' }, svg)];
          const vals = pos.map((column, li) => column.map((p) => {
            h.el('circle', { cx: p.x, cy: p.y, r: 24, fill: 'var(--surface)', stroke: li === 0 ? 'var(--ink)' : 'var(--accent)', 'stroke-width': 1.8 }, ng[li]);
            return txt(h, ng[li], p.x, p.y, '', { size: 12, mono: true, mid: true });
          }));
          txt(h, svg, 60, 50, 'Entrée x', { size: 13, fill: 'var(--muted)' });
          txt(h, ng[1], 195, 50, 'Cachée (ReLU)', { size: 13, fill: 'var(--muted)' });
          txt(h, ng[2], 330, 50, 'Sortie (linéaire)', { size: 13, fill: 'var(--muted)' });

          const b1 = h.el('g', { style: 'transition:opacity .5s' }, svg);
          const b2 = h.el('g', { style: 'transition:opacity .5s' }, svg);
          const hl1 = h.el('rect', { x: 368, y: 48, width: 330, height: 106, rx: 8, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, opacity: 0, style: 'transition:opacity .4s' }, svg);
          const hl2 = h.el('rect', { x: 368, y: 208, width: 330, height: 94, rx: 8, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, opacity: 0, style: 'transition:opacity .4s' }, svg);
          txt(h, b1, 380, 66, 'Couche cachée : z = W·x puis a = ReLU(z)', { size: 12, fill: 'var(--muted)', anchor: 'start' });
          const mW1 = mat(h, b1, 380, 82, 3, 2, 40, 22, 1);
          txt(h, b1, 472, 115, '·', { size: 18, mid: true });
          const mX = mat(h, b1, 484, 82, 2, 1, 42, 22, 2);
          mX.move(484, 93);
          txt(h, b1, 544, 115, '=', { size: 16, mid: true });
          const mZ = mat(h, b1, 560, 82, 3, 1, 42, 22, 2);
          h.el('line', { x1: 610, y1: 115, x2: 640, y2: 115, stroke: 'var(--accent)', 'stroke-width': 2, 'marker-end': 'url(#arrow-accent)' }, b1);
          txt(h, b1, 625, 103, 'ReLU', { size: 11, fill: 'var(--accent)' });
          const mA = mat(h, b1, 650, 82, 3, 1, 42, 22, 2);
          txt(h, b2, 380, 226, 'Sortie : y = W·a', { size: 12, fill: 'var(--muted)', anchor: 'start' });
          const mW2 = mat(h, b2, 380, 244, 2, 3, 40, 22, 1);
          txt(h, b2, 512, 266, '·', { size: 18, mid: true });
          const mA2 = mat(h, b2, 524, 233, 3, 1, 42, 22, 2);
          mA2.move(524, 233);
          txt(h, b2, 582, 266, '=', { size: 16, mid: true });
          const mY = mat(h, b2, 598, 244, 2, 1, 42, 22, 2);
          mW1.set(W1);
          mW2.set(W2);
          txt(h, svg, 360, 402, 'Arêtes vertes : poids positifs, rouges : poids négatifs.', { size: 12, fill: 'var(--muted)' });

          const refresh = () => {
            const r = forward();
            X.forEach((v, i) => { vals[0][i].textContent = h.fmt(v, 2); });
            r.a.forEach((v, i) => { vals[1][i].textContent = h.fmt(v, 2); });
            r.y.forEach((v, i) => { vals[2][i].textContent = h.fmt(v, 2); });
            mX.set(col(X));
            mZ.set(col(r.z));
            mA.set(col(r.a));
            mA2.set(col(r.a));
            mY.set(col(r.y));
          };

          const flow = (list) => {
            const dots = list.map((l) => h.el('circle', { r: 4.5, fill: 'var(--accent)', cx: l.getAttribute('x1'), cy: l.getAttribute('y1') }, svg));
            return h.tween(800, (p) => {
              dots.forEach((d, i) => {
                const l = list[i];
                d.setAttribute('cx', h.lerp(+l.getAttribute('x1'), +l.getAttribute('x2'), p));
                d.setAttribute('cy', h.lerp(+l.getAttribute('y1'), +l.getAttribute('y2'), p));
              });
            }, h.ease.linear).then(() => dots.forEach((d) => d.remove()));
          };

          let run = 0;
          const setOp = (el, v) => { el.style.opacity = v; };
          const play = async () => {
            const id = ++run;
            [eg[0], eg[1], ng[1], ng[2], b1, b2].forEach((e) => setOp(e, 0));
            hl1.setAttribute('opacity', 0);
            hl2.setAttribute('opacity', 0);
            await h.sleep(500);
            if (id !== run) return;
            setOp(eg[0], 1);
            hl1.setAttribute('opacity', 1);
            setOp(b1, 1);
            await flow(edgeLines[0]);
            if (id !== run) return;
            setOp(ng[1], 1);
            await h.sleep(900);
            if (id !== run) return;
            hl1.setAttribute('opacity', 0);
            hl2.setAttribute('opacity', 1);
            setOp(eg[1], 1);
            setOp(b2, 1);
            await flow(edgeLines[1]);
            if (id !== run) return;
            setOp(ng[2], 1);
          };

          h.slider({ label: '$x_1$', min: -1, max: 1, step: 0.1, value: X[0], decimals: 1, onInput: (v) => { X[0] = v; refresh(); } });
          h.slider({ label: '$x_2$', min: -1, max: 1, step: 0.1, value: X[1], decimals: 1, onInput: (v) => { X[1] = v; refresh(); } });
          h.button('↻ Rejouer', play, 'primary');
          refresh();
          play();
        }
      },
      {
        title: 'Non-linéarités',
        duration: 12000,
        text: String.raw`<p>La fonction $g$ doit être <b>dérivable</b> pour pouvoir entraîner le réseau par descente de gradient. Trois choix classiques :</p>
<ul>
<li><b>sigmoïde</b> $\sigma(x)=\dfrac{1}{1+e^{-x}}$, sortie dans $]0,1[$ ;</li>
<li><b>tanh</b>, sortie dans $]-1,1[$ ;</li>
<li><b>ReLU</b> $\max(0,x)$ [Glorot, 2011] : très simple, dérivée 0 ou 1.</li>
</ul>
<p>Faites glisser le point sur la courbe : la courbe pleine est $g(x)$, la pointillée est sa dérivée $g'(x)$.</p>`,
        note: String.raw`Pour sigmoïde et tanh, quand $|x|$ est grand la courbe est plate : $g'(x)\approx 0$ (zone orange). La fonction est <b>saturée</b> et le gradient quasi nul.`,
        check: {
          q: String.raw`Pour quelle valeur de $x$ la dérivée de ReLU vaut-elle 1 ?`,
          choices: ['Pour tout $x$', 'Pour $x > 0$', 'Seulement en $x = 0$'],
          answer: 1,
          explain: String.raw`ReLU vaut $x$ pour $x>0$ (pente 1) et $0$ pour $x<0$ (pente 0).`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const b = { x0: 70, x1: 690, y0: 30, y1: 350 };
          const sx = h.scale(-6, 6, b.x0, b.x1);
          const YMIN = -1.25, YMAX = 1.5;
          const sy = h.scale(YMIN, YMAX, b.y1, b.y0);
          const clamp = (v) => Math.max(YMIN, Math.min(YMAX, v));
          const state = { fn: 'sigmoid', x: 1.2 };
          const base = h.el('g', {}, svg);
          axes(h, base, b, sx, sy, { xt: [-6, -4, -2, 0, 2, 4, 6], yt: [-1, -0.5, 0, 0.5, 1], grid: true, xl: 'x', yl: 'valeur' });
          h.el('line', { x1: b.x0, y1: sy(0), x2: b.x1, y2: sy(0), stroke: 'var(--muted)', 'stroke-width': 1 }, base);
          h.el('line', { x1: sx(0), y1: b.y0, x2: sx(0), y2: b.y1, stroke: 'var(--muted)', 'stroke-width': 1 }, base);
          const dyn = h.el('g', {}, svg);
          const out = h.readout('');

          const path = (f) => {
            let d = '';
            for (let i = 0; i <= 240; i++) {
              const x = -6 + 12 * i / 240;
              d += (i ? 'L' : 'M') + sx(x).toFixed(1) + ',' + sy(clamp(f(x))).toFixed(1);
            }
            return d;
          };

          const render = () => {
            const f = FN[state.fn];
            while (dyn.firstChild) dyn.removeChild(dyn.firstChild);
            if (f.sat) {
              let xs = 0;
              while (f.d(xs) > 0.05 && xs < 6) xs += 0.01;
              h.el('rect', { x: sx(xs), y: b.y0, width: b.x1 - sx(xs), height: b.y1 - b.y0, fill: 'var(--warn)', opacity: 0.13 }, dyn);
              h.el('rect', { x: b.x0, y: b.y0, width: sx(-xs) - b.x0, height: b.y1 - b.y0, fill: 'var(--warn)', opacity: 0.13 }, dyn);
              txt(h, dyn, (sx(xs) + b.x1) / 2, b.y0 + 80, "g′ ≈ 0", { size: 13, fill: 'var(--warn)', bold: true });
              txt(h, dyn, (b.x0 + sx(-xs)) / 2, b.y0 + 80, "g′ ≈ 0", { size: 13, fill: 'var(--warn)', bold: true });
            }
            h.el('path', { d: path(f.g), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3 }, dyn);
            h.el('path', { d: path(f.d), fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'stroke-dasharray': '7 5' }, dyn);
            h.el('line', { x1: b.x0 + 12, y1: b.y0 + 16, x2: b.x0 + 40, y2: b.y0 + 16, stroke: 'var(--accent)', 'stroke-width': 3 }, dyn);
            txt(h, dyn, b.x0 + 48, b.y0 + 16, "g(x)", { size: 13, anchor: 'start', mid: true });
            h.el('line', { x1: b.x0 + 12, y1: b.y0 + 36, x2: b.x0 + 40, y2: b.y0 + 36, stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'stroke-dasharray': '7 5' }, dyn);
            txt(h, dyn, b.x0 + 48, b.y0 + 36, "g′(x)", { size: 13, anchor: 'start', mid: true });
            const gx = f.g(state.x), dx = f.d(state.x);
            h.el('line', { x1: sx(state.x), y1: b.y0, x2: sx(state.x), y2: b.y1, stroke: 'var(--ink)', 'stroke-dasharray': '3 4', opacity: 0.6 }, dyn);
            h.el('circle', { cx: sx(state.x), cy: sy(clamp(gx)), r: 7, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2 }, dyn);
            h.el('circle', { cx: sx(state.x), cy: sy(clamp(dx)), r: 7, fill: 'var(--accent-2)', stroke: 'var(--surface)', 'stroke-width': 2 }, dyn);
            out(`${f.name} : $x=${h.fmt(state.x, 2)}$ &nbsp; $g(x)=${h.fmt(gx, 3)}$ &nbsp; $g'(x)=${h.fmt(dx, 3)}$`);
          };

          const sl = h.slider({ label: 'x', min: -6, max: 6, step: 0.05, value: state.x, decimals: 2, onInput: (v) => { state.x = v; render(); } });
          h.toggle([['sigmoid', 'Sigmoïde'], ['tanh', 'tanh'], ['relu', 'ReLU']], state.fn, (v) => { state.fn = v; render(); });
          const hit = h.el('rect', { x: b.x0, y: b.y0, width: b.x1 - b.x0, height: b.y1 - b.y0, fill: 'transparent', style: 'cursor:ew-resize' }, svg);
          let drag = false;
          const move = (e) => {
            const p = svgPoint(svg, e);
            state.x = Math.max(-6, Math.min(6, sx.invert(p.x)));
            sl.set(state.x);
            render();
          };
          hit.addEventListener('pointerdown', (e) => { drag = true; move(e); });
          const onMove = (e) => { if (drag) move(e); };
          const onUp = () => { drag = false; };
          window.addEventListener('pointermove', onMove);
          window.addEventListener('pointerup', onUp);
          h.onLeave(() => { window.removeEventListener('pointermove', onMove); window.removeEventListener('pointerup', onUp); });
          render();
        }
      },
      {
        title: 'Gradient qui s\'évanouit',
        duration: 10000,
        text: String.raw`<p>Lors de l'apprentissage, le gradient traverse toutes les couches : à chaque couche il est <b>multiplié par la dérivée</b> $g'(z)$.</p>
<p>La dérivée de la sigmoïde vaut au plus $0{,}25$. Après $k$ couches : au plus $0{,}25^k$ ! Le gradient tend vers 0 et les premières couches n'apprennent presque plus : c'est le <b>gradient qui s'évanouit</b> (<i>vanishing gradient</i>).</p>
<p>ReLU a une dérivée égale à 1 pour $z>0$ : le gradient passe sans s'affaiblir. C'est l'une des raisons de son succès.</p>`,
        note: String.raw`Un neurone saturé ($g'(z)\approx 0$) bloque le gradient. Avec ReLU, un neurone à $z<0$ bloque aussi (dérivée 0), mais ceux à $z>0$ transmettent tout.`,
        draw(h) {
          const svg = h.svg(720, 420);
          const L = 8;
          const b = { x0: 90, x1: 690, y0: 40, y1: 330 };
          const sy = h.scale(-8, 0, b.y1, b.y0);
          const base = h.el('g', {}, svg);
          axes(h, base, b, (v) => v, sy, {
            xt: [], yt: [0, -2, -4, -6, -8], grid: true, yl: 'gradient restant (échelle log)',
            yf: (v) => ({ '0': '1', '-2': '10⁻²', '-4': '10⁻⁴', '-6': '10⁻⁶', '-8': '10⁻⁸' })[v]
          });
          txt(h, base, (b.x0 + b.x1) / 2, b.y1 + 40, 'nombre de couches traversées par le gradient', { size: 13, fill: 'var(--muted)' });
          const step = (b.x1 - b.x0) / L;
          const bars = [], labels = [];
          for (let k = 1; k <= L; k++) {
            const x = b.x0 + step * (k - 0.5);
            bars.push(h.el('rect', { x: x - 20, y: b.y1, width: 40, height: 0, rx: 3, fill: 'var(--accent-2)' }, svg));
            labels.push(txt(h, svg, x, b.y1 - 8, '', { size: 12, mono: true }));
            txt(h, svg, x, b.y1 + 19, String(k), { size: 12, fill: 'var(--muted)' });
          }
          const state = { fn: 'sigmoid', z: 0 };
          const cur = new Array(L).fill(0);
          const out = h.readout('');
          let seq = 0;
          const sci = (v) => (v === 0 ? '0' : v >= 0.01 ? h.fmt(v, 2) : v.toExponential(0).replace('e-', 'e−'));
          const target = () => {
            const d = FN[state.fn].d(state.z);
            return Array.from({ length: L }, (_, k) => {
              const v = Math.pow(d, k + 1);
              return { v, lg: v <= 1e-8 ? -8 : Math.max(-8, Math.log10(v)) };
            });
          };
          const apply = (arr) => {
            arr.forEach((lg, k) => {
              const y = sy(lg);
              bars[k].setAttribute('y', y);
              bars[k].setAttribute('height', Math.max(0, b.y1 - y));
              labels[k].setAttribute('y', Math.min(b.y1 - 8, y - 8));
            });
          };
          const update = async (animate) => {
            const t = target();
            const d = FN[state.fn].d(state.z);
            t.forEach((e, k) => { labels[k].textContent = sci(e.v); });
            out(`$g'(z)=${h.fmt(d, 3)}$ &nbsp; après ${L} couches : ${sci(t[L - 1].v)}`);
            const from = cur.slice();
            const to = t.map((e) => (e.v === 0 ? -8 : e.lg));
            const id = ++seq;
            if (!animate) { to.forEach((v, k) => { cur[k] = v; }); apply(cur); return; }
            await h.tween(350, (p) => {
              if (id !== seq) return;
              to.forEach((v, k) => { cur[k] = h.lerp(from[k], v, p); });
              apply(cur);
            });
          };
          h.toggle([['sigmoid', 'Sigmoïde'], ['tanh', 'tanh'], ['relu', 'ReLU']], state.fn, (v) => { state.fn = v; update(true); });
          h.slider({ label: 'z', min: -6, max: 6, step: 0.1, value: 0, decimals: 1, onInput: (v) => { state.z = v; update(true); } });
          update(false).then(() => {});
          cur.fill(-8);
          apply(cur);
          h.after(300, () => update(true));
        }
      },
      {
        title: 'Sans non-linéarité : rien',
        duration: 12000,
        text: String.raw`<p>Que se passe-t-il si on empile des couches <b>sans</b> fonction d'activation ? Deux couches donnent $y = W_2(W_1 x)$.</p>
<p>Or le produit de matrices est associatif : $y = (W_2 W_1)\,x = W\,x$. Les deux couches <b>se réduisent à une seule</b> matrice $W = W_2W_1$ : on n'a gagné aucune puissance d'expression.</p>
<p>Avec une non-linéarité entre les couches ($W_2\,\mathrm{ReLU}(W_1x)$), ce n'est plus vrai : aucune matrice unique ne reproduit le résultat. Comparez avec les curseurs.</p>`,
        note: String.raw`Un réseau profond sans non-linéarité n'est qu'un modèle linéaire déguisé. Ce sont les non-linéarités qui rendent la profondeur utile.`,
        check: {
          q: 'On empile 5 couches entièrement connectées sans aucune fonction d\'activation. Le réseau est équivalent à :',
          choices: ['Un réseau non linéaire à 5 couches', 'Une seule transformation linéaire', 'Un réseau à 5 fois plus de paramètres utiles'],
          answer: 1,
          explain: String.raw`Le produit $W_5\cdots W_2W_1$ est une seule matrice : le modèle reste linéaire.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const W1 = [[1, -1], [0.5, 1]];
          const W2 = [[1, 0.5], [-1, 1]];
          const P = [
            [W2[0][0] * W1[0][0] + W2[0][1] * W1[1][0], W2[0][0] * W1[0][1] + W2[0][1] * W1[1][1]],
            [W2[1][0] * W1[0][0] + W2[1][1] * W1[1][0], W2[1][0] * W1[0][1] + W2[1][1] * W1[1][1]]
          ];
          const X = [1, -1];
          const state = { mode: 'lin' };
          const g = h.el('g', {}, svg);
          const res = h.el('g', {}, svg);
          const t1 = txt(h, res, 70, 285, '', { size: 14, mono: true, anchor: 'start' });
          const t2 = txt(h, res, 70, 318, '', { size: 14, mono: true, anchor: 'start' });
          const t3 = txt(h, res, 70, 358, '', { size: 14, bold: true, anchor: 'start' });
          let build = null;
          const apply = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
          const vec = (v) => '[' + h.fmt(v[0], 2) + ' ; ' + h.fmt(v[1], 2) + ']';
          const refresh = () => {
            const two = state.mode === 'lin'
              ? apply(W2, apply(W1, X))
              : apply(W2, apply(W1, X).map(relu));
            const one = apply(P, X);
            t1.textContent = (state.mode === 'lin' ? 'W₂·(W₁·x) = ' : 'W₂·ReLU(W₁·x) = ') + vec(two);
            t2.textContent = '(W₂·W₁)·x = ' + vec(one);
            const same = Math.abs(two[0] - one[0]) < 1e-9 && Math.abs(two[1] - one[1]) < 1e-9;
            t3.textContent = same ? 'Identiques : deux couches linéaires = une seule couche.' : 'Différents : la non-linéarité empêche la fusion en une matrice.';
            t3.setAttribute('fill', same ? 'var(--ok)' : 'var(--bad)');
          };
          let run = 0;
          const collapse = async () => {
            const id = ++run;
            build();
            await h.sleep(900);
            if (id !== run || state.mode !== 'lin') return;
            const els = build.els;
            await h.tween(1100, (p) => {
              els.w1.move(h.lerp(200, 70, p), 100);
              els.w1.g.style.opacity = 1 - p;
              els.w2.g.style.opacity = 1 - p;
              els.p.g.style.opacity = p;
              els.d1.style.opacity = 1 - p;
              els.d2.setAttribute('x', h.lerp(310, 180, p));
              els.x.move(h.lerp(330, 200, p), 100);
              els.cap.style.opacity = p;
            });
          };
          build = () => {
            while (g.firstChild) g.removeChild(g.firstChild);
            const els = {};
            txt(h, g, 40, 136, 'y =', { size: 16, mid: true });
            if (state.mode === 'lin') {
              els.w2 = mat(h, g, 70, 100, 2, 2, 44, 36, 2, 'W₂');
              els.w2.set(W2);
              els.d1 = txt(h, g, 180, 136, '·', { size: 22, mid: true });
              els.w1 = mat(h, g, 200, 100, 2, 2, 44, 36, 2, 'W₁');
              els.w1.set(W1);
              els.d2 = txt(h, g, 310, 136, '·', { size: 22, mid: true });
              els.x = mat(h, g, 330, 100, 2, 1, 44, 36, 1, 'x');
              els.x.set(col(X));
              els.p = mat(h, g, 70, 100, 2, 2, 44, 36, 2, 'W = W₂W₁');
              els.p.set(P);
              els.p.g.style.opacity = 0;
              els.cap = txt(h, g, 160, 205, 'une seule couche équivalente', { size: 13, fill: 'var(--accent)', bold: true });
              els.cap.style.opacity = 0;
            } else {
              const w2 = mat(h, g, 70, 100, 2, 2, 44, 36, 2, 'W₂');
              w2.set(W2);
              txt(h, g, 178, 136, '·', { size: 22, mid: true });
              txt(h, g, 190, 136, 'ReLU(', { size: 16, anchor: 'start', mid: true, fill: 'var(--accent)', bold: true });
              const w1 = mat(h, g, 252, 100, 2, 2, 44, 36, 2, 'W₁');
              w1.set(W1);
              txt(h, g, 352, 136, '·', { size: 22, mid: true });
              const x = mat(h, g, 366, 100, 2, 1, 44, 36, 1, 'x');
              x.set(col(X));
              txt(h, g, 428, 136, ')', { size: 16, anchor: 'start', mid: true, fill: 'var(--accent)', bold: true });
              txt(h, g, 300, 205, 'impossible de fusionner en une matrice', { size: 13, fill: 'var(--bad)', bold: true });
            }
            build.els = els;
            refresh();
          };
          const wrap = h.toggle([['lin', 'Sans activation'], ['relu', 'Avec ReLU entre les couches']], 'lin', (v) => { state.mode = v; if (v === 'lin') collapse(); else { run++; build(); } });
          h.slider({ label: '$x_1$', min: -2, max: 2, step: 0.1, value: X[0], decimals: 1, onInput: (v) => { X[0] = v; if (build.els.x) build.els.x.set(col(X)); if (state.mode === 'relu') build(); refresh(); } });
          h.slider({ label: '$x_2$', min: -2, max: 2, step: 0.1, value: X[1], decimals: 1, onInput: (v) => { X[1] = v; if (build.els.x) build.els.x.set(col(X)); if (state.mode === 'relu') build(); refresh(); } });
          h.button('▶ Fusionner', () => { pick(wrap, 0); }, 'primary');
          build();
          collapse();
        }
      },
      {
        title: 'Le cas XOR',
        duration: 13000,
        text: String.raw`<p>Le XOR vaut 1 quand <b>exactement une</b> des deux entrées vaut 1. Aucune droite ne sépare les points $(0,1),(1,0)$ de $(0,0),(1,1)$ : un neurone seul échoue.</p>
<p>Un MLP 2-2-1 avec des poids choisis à la main y arrive :</p>
<p>$h_1=\mathrm{ReLU}(x_1+x_2)$, &nbsp; $h_2=\mathrm{ReLU}(x_1+x_2-1)$, &nbsp; $\hat y = h_1-2h_2$.</p>
<p>La couche cachée <b>déforme l'espace</b> : dans $(h_1,h_2)$, les deux points de classe 1 se confondent en $(1,0)$ et une droite suffit. Regardez les points se déplacer.</p>`,
        note: String.raw`Rôle de la couche cachée : trouver une représentation des données dans laquelle le problème devient <b>linéairement séparable</b>. La dernière couche n'a plus qu'à tracer une droite.`,
        check: {
          q: 'Pourquoi un neurone seul ne peut-il pas résoudre le XOR ?',
          choices: ['Il n\'a pas assez d\'entrées', 'Les deux classes ne sont pas séparables par une droite', 'Il n\'a pas de biais'],
          answer: 1,
          explain: String.raw`Un neurone trace une frontière linéaire ; les points XOR de classe 1 sont aux deux coins opposés d'un carré.`
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
          const hid = (x1, x2) => [relu(x1 + x2), relu(x1 + x2 - 1)];
          const pos = layout([2, 2, 1], 38, 200, 120, 280, 90);
          txt(h, svg, 119, 80, 'Réseau 2-2-1', { size: 13, fill: 'var(--muted)' });
          const wts = [[[1, 1], [1, 1]], [[1, -2]]];
          wts.forEach((W, li) => pos[li].forEach((a, i) => pos[li + 1].forEach((b, j) => {
            const w = W[j][i];
            h.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': 1 + 1.5 * Math.abs(w), opacity: 0.8 }, svg);
          })));
          const names = [['x₁', 'x₂'], ['h₁', 'h₂'], ['ŷ']];
          pos.forEach((column, li) => column.forEach((p, j) => {
            h.el('circle', { cx: p.x, cy: p.y, r: 17, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            txt(h, svg, p.x, p.y, names[li][j], { size: 12, mid: true });
          }));
          h.label(svg, 8, 300, 230, 100, String.raw`<div style="font-size:13px">$h_1=\mathrm{ReLU}(x_1+x_2)$<br>$h_2=\mathrm{ReLU}(x_1+x_2-1)$<br>$\hat y = h_1-2h_2$ &nbsp; (seuil 0,5)</div>`);

          const bl = { x0: 268, x1: 458, y0: 100, y1: 290 };
          const br = { x0: 520, x1: 700, y0: 100, y1: 290 };
          const slx = h.scale(-0.5, 1.5, bl.x0, bl.x1), sly = h.scale(-0.5, 1.5, bl.y1, bl.y0);
          const srx = h.scale(-0.5, 2.5, br.x0, br.x1), sry = h.scale(-0.5, 1.5, br.y1, br.y0);
          txt(h, svg, (bl.x0 + bl.x1) / 2, 80, "Espace d'entrée", { size: 13, bold: true });
          txt(h, svg, (br.x0 + br.x1) / 2, 80, 'Espace caché', { size: 13, bold: true });
          axes(h, svg, bl, slx, sly, { xt: [0, 1], yt: [0, 1], grid: true, xl: 'x₁', yl: 'x₂' });
          axes(h, svg, br, srx, sry, { xt: [0, 1, 2], yt: [0, 1], grid: true, xl: 'h₁', yl: 'h₂' });
          const tries = h.el('g', { style: 'transition:opacity .5s' }, svg);
          h.el('line', { x1: slx(-0.5), y1: sly(1.0), x2: slx(1.5), y2: sly(0), stroke: 'var(--muted)', 'stroke-width': 2, 'stroke-dasharray': '6 5' }, tries);
          h.el('line', { x1: slx(-0.5), y1: sly(0.5), x2: slx(1.5), y2: sly(0.5), stroke: 'var(--muted)', 'stroke-width': 2, 'stroke-dasharray': '6 5' }, tries);
          txt(h, svg, (bl.x0 + bl.x1) / 2, 328, 'Aucune droite ne sépare', { size: 12, fill: 'var(--bad)' });
          txt(h, svg, (bl.x0 + bl.x1) / 2, 345, 'les deux classes.', { size: 12, fill: 'var(--bad)' });
          const dec = h.el('line', { x1: srx(-0.5), y1: sry(-0.5), x2: srx(2.5), y2: sry(1), stroke: 'var(--accent)', 'stroke-width': 2.5, opacity: 0, style: 'transition:opacity .6s' }, svg);
          const verdict = [txt(h, svg, (br.x0 + br.x1) / 2, 328, 'Une droite suffit :', { size: 12, fill: 'var(--ok)', opacity: 0 }), txt(h, svg, (br.x0 + br.x1) / 2, 345, 'h₁ − 2h₂ = 0,5', { size: 12, fill: 'var(--ok)', opacity: 0 })];
          verdict.forEach((v) => { v.style.transition = 'opacity .6s'; });
          data.forEach(([x1, x2, c]) => {
            h.el('circle', { cx: slx(x1), cy: sly(x2), r: 10, fill: c ? 'var(--pos)' : 'var(--neg)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            txt(h, svg, slx(x1) + (x1 ? -14 : 14), sly(x2) + (x2 ? 22 : -16), `(${x1},${x2})`, { size: 11, fill: 'var(--muted)' });
          });
          const pts = data.map(([x1, x2, c]) => ({
            c: h.el('circle', { r: 10, fill: c ? 'var(--pos)' : 'var(--neg)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg),
            from: [x1, x2], to: hid(x1, x2)
          }));
          const place = (p) => pts.forEach((o) => {
            o.c.setAttribute('cx', srx(h.lerp(o.from[0], o.to[0], p)));
            o.c.setAttribute('cy', sry(h.lerp(o.from[1], o.to[1], p)));
          });
          h.el('circle', { cx: 285, cy: 398, r: 7, fill: 'var(--pos)', stroke: 'var(--ink)' }, svg);
          txt(h, svg, 298, 398, 'XOR = 1', { size: 12, anchor: 'start', mid: true });
          h.el('circle', { cx: 380, cy: 398, r: 7, fill: 'var(--neg)', stroke: 'var(--ink)' }, svg);
          txt(h, svg, 393, 398, 'XOR = 0', { size: 12, anchor: 'start', mid: true });
          h.readout(String.raw`$(0,0)\to(0,0)\to 0$ &nbsp; $(0,1)\to(1,0)\to 1$ &nbsp; $(1,0)\to(1,0)\to 1$ &nbsp; $(1,1)\to(2,1)\to 0$`);

          let run = 0;
          const play = async () => {
            const id = ++run;
            place(0);
            dec.setAttribute('opacity', 0);
            verdict.forEach((v) => v.setAttribute('opacity', 0));
            await h.sleep(1100);
            if (id !== run) return;
            await h.tween(1800, (p) => { if (id === run) place(p); });
            if (id !== run) return;
            dec.setAttribute('opacity', 1);
            verdict.forEach((v) => v.setAttribute('opacity', 1));
          };
          h.button('↻ Rejouer', play, 'primary');
          play();
        }
      },
      {
        title: 'Architecture d\'un MLP',
        duration: 12000,
        text: String.raw`<p>La nature du problème fixe la <b>sortie</b> du réseau :</p>
<ul>
<li><b>entrée</b> : autant de neurones que la dimension $d$ des données ;</li>
<li><b>sortie</b> : 1 neurone linéaire (régression), 1 neurone sigmoïde (classification binaire) ou $K$ neurones softmax (multiclasse) ;</li>
<li><b>couches cachées</b> : nombre et taille à déterminer (expertise, validation croisée, architectures connues).</li>
</ul>
<p>Changez le type de problème et les tailles : la sortie, son activation et la fonction de coût s'adaptent.</p>`,
        note: String.raw`Le choix de la fonction d'activation de sortie et de la fonction de coût dépend du problème, pas des couches cachées (en général ReLU).`,
        check: {
          q: 'Classification de chiffres manuscrits en 10 classes exclusives. Quelle couche de sortie ?',
          choices: ['1 neurone linéaire', '1 neurone sigmoïde', '10 neurones avec softmax'],
          answer: 2,
          explain: String.raw`Une probabilité par classe, qui somment à 1 : softmax sur $K=10$ neurones, avec entropie croisée catégorielle.`
        },
        draw(h) {
          const svg = h.svg(720, 440);
          const K = 3;
          const state = { prob: 'reg', layers: 2, size: 4, d: 4 };
          const g = h.el('g', {}, svg);
          const cards = {
            reg: { out: '1 neurone', act: String.raw`Linéaire : $\hat y = z$`, loss: 'MSE ou MAE', tag: 'ŷ ∈ ℝ', color: 'var(--accent-2)' },
            bin: { out: '1 neurone', act: String.raw`Sigmoïde : $\hat p=\sigma(z)$`, loss: 'Entropie croisée binaire', tag: 'p ∈ ]0,1[', color: 'var(--accent-2)' },
            multi: { out: K + ' neurones (K classes)', act: String.raw`Softmax : $p_k=\dfrac{e^{z_k}}{\sum_j e^{z_j}}$`, loss: 'Entropie croisée catégorielle', tag: null, color: 'var(--accent-2)' }
          };
          const render = () => {
            while (g.firstChild) g.removeChild(g.firstChild);
            const c = cards[state.prob];
            const outN = state.prob === 'multi' ? K : 1;
            const sizes = [state.d].concat(new Array(state.layers).fill(state.size), [outN]);
            const pos = layout(sizes, 50, 380, 70, 400, 50);
            for (let li = 0; li < sizes.length - 1; li++) {
              pos[li].forEach((a) => pos[li + 1].forEach((b) => h.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'var(--muted)', 'stroke-width': 1, opacity: 0.55 }, g)));
            }
            pos.forEach((column, li) => column.forEach((p) => {
              const out = li === sizes.length - 1;
              h.el('circle', { cx: p.x, cy: p.y, r: 15, fill: out ? 'var(--accent-2)' : 'var(--surface)', 'fill-opacity': out ? 0.3 : 1, stroke: out ? 'var(--accent-2)' : li === 0 ? 'var(--ink)' : 'var(--accent)', 'stroke-width': 1.8 }, g);
            }));
            txt(h, g, 50, 42, `Entrée : d = ${state.d}`, { size: 13, fill: 'var(--muted)', anchor: 'start' });
            txt(h, g, (50 + 380) / 2, 42, `Cachées (ReLU) : ${state.layers} × ${state.size}`, { size: 13, fill: 'var(--muted)' });
            txt(h, g, 380, 42, 'Sortie', { size: 13, fill: 'var(--accent-2)', bold: true, anchor: 'end' });
            const last = pos[pos.length - 1];
            last.forEach((p, j) => txt(h, g, p.x + 24, p.y, c.tag || 'p' + '₁₂₃'[j], { size: 13, anchor: 'start', mid: true, fill: 'var(--accent-2)', bold: true }));
            const rows = [
              ['Couche de sortie', c.out],
              ['Activation de sortie', c.act],
              ['Fonction de coût', c.loss]
            ];
            rows.forEach(([title, body], i) => {
              const y = 70 + i * 112;
              h.el('rect', { x: 492, y, width: 216, height: 96, rx: 10, fill: 'var(--surface-2)', stroke: 'var(--line)' }, g);
              h.el('rect', { x: 492, y, width: 5, height: 96, rx: 2, fill: i === 0 ? 'var(--accent)' : i === 1 ? 'var(--accent-2)' : 'var(--ok)' }, g);
              h.label(g, 506, y + 6, 196, 86, `<div style="font-size:12px;color:var(--muted)">${title}</div><div style="font-size:14px;margin-top:6px"><b>${body}</b></div>`);
            });
          };
          h.toggle([['reg', 'Régression'], ['bin', 'Classif. binaire'], ['multi', 'Multiclasse']], state.prob, (v) => { state.prob = v; render(); });
          h.slider({ label: 'd', min: 2, max: 6, step: 1, value: state.d, decimals: 0, onInput: (v) => { state.d = v; render(); } });
          h.slider({ label: 'Couches', min: 1, max: 3, step: 1, value: state.layers, decimals: 0, onInput: (v) => { state.layers = v; render(); } });
          h.slider({ label: 'Taille', min: 2, max: 6, step: 1, value: state.size, decimals: 0, onInput: (v) => { state.size = v; render(); } });
          render();
        }
      }
    ]
  });
})();
