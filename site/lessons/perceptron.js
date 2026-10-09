/**
 * Lesson: biological neuron, artificial neuron, perceptron as a linear classifier,
 * Rosenblatt's learning algorithm, convergence, and the XOR limitation.
 */
(function () {
  let uid = 0;
  const SUB = ['', '₁', '₂', '₃', '₄'];

  /**
   * Formats a number with a French decimal comma and a true minus sign.
   */
  function num(h, v, d) {
    return h.fmt(v, d).replace('-', '−');
  }

  /**
   * Builds the shared, linearly separable 2D dataset (22 points, 11 per class, with a margin).
   */
  function makeDataset() {
    let s = 20240611;
    const rnd = () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
    const wt = [1, 0.7], bt = -0.3, nw = Math.hypot(wt[0], wt[1]);
    const pts = [];
    const count = { 1: 0, '-1': 0 };
    let guard = 0;
    while (pts.length < 22 && guard++ < 20000) {
      const x = (rnd() * 2 - 1) * 3.6, y = (rnd() * 2 - 1) * 3.6;
      const sd = (wt[0] * x + wt[1] * y + bt) / nw;
      if (Math.abs(sd) < 0.8) continue;
      const c = sd > 0 ? 1 : -1;
      if (count[c] >= 11) continue;
      count[c]++;
      pts.push({ x, y, c, s: sd });
    }
    return pts;
  }

  /**
   * Returns a copy of the dataset where the two points farthest from the true line are flipped.
   */
  function makeNoisy(pts) {
    const copy = pts.map((p) => Object.assign({}, p));
    const order = copy.map((p, i) => i).sort((a, b) => Math.abs(copy[b].s) - Math.abs(copy[a].s));
    let flippedPos = false, flippedNeg = false;
    for (const i of order) {
      if (copy[i].c > 0 && !flippedPos) { copy[i].c = -1; flippedPos = true; }
      else if (copy[i].c < 0 && !flippedNeg) { copy[i].c = 1; flippedNeg = true; }
      if (flippedPos && flippedNeg) break;
    }
    return copy;
  }

  /**
   * Pre-activation of a point for weights w and bias b.
   */
  function score(w, b, p) {
    return w[0] * p.x + w[1] * p.y + b;
  }

  /**
   * True when the perceptron rule would update on this point (y * a <= 0).
   */
  function isErr(w, b, p) {
    return p.c * score(w, b, p) <= 0;
  }

  /**
   * Number of points on which the perceptron rule would update.
   */
  function countErr(w, b, pts) {
    let k = 0;
    for (const p of pts) if (isErr(w, b, p)) k++;
    return k;
  }

  /**
   * Colour token of a class label.
   */
  function colorOf(c) {
    return c > 0 ? 'var(--pos)' : 'var(--neg)';
  }

  /**
   * Runs the perceptron rule epoch by epoch and returns a snapshot after each epoch.
   */
  function trainTrace(pts, eta, maxEpochs) {
    let w = [0.5, -0.9], b = 0.5;
    const snaps = [{ w: w.slice(), b, err: null }];
    for (let e = 0; e < maxEpochs; e++) {
      let errs = 0;
      for (const p of pts) {
        if (p.c * score(w, b, p) <= 0) {
          w = [w[0] + eta * p.c * p.x, w[1] + eta * p.c * p.y];
          b += eta * p.c;
          errs++;
        }
      }
      snaps.push({ w: w.slice(), b, err: errs });
      if (errs === 0) break;
    }
    return snaps;
  }

  /**
   * Clips the square [lo,hi]^2 against the half-plane sg*(w.p + b) >= 0; returns polygon and cut points.
   */
  function clipHalf(lo, hi, w, b, sg) {
    const f = (p) => sg * (w[0] * p[0] + w[1] * p[1] + b);
    const box = [[lo, lo], [hi, lo], [hi, hi], [lo, hi]];
    const poly = [], cut = [];
    for (let i = 0; i < 4; i++) {
      const c = box[i], n = box[(i + 1) % 4], fc = f(c), fn = f(n);
      if (fc >= 0) poly.push(c);
      if ((fc >= 0) !== (fn >= 0)) {
        const t = fc / (fc - fn);
        const q = [c[0] + (n[0] - c[0]) * t, c[1] + (n[1] - c[1]) * t];
        poly.push(q);
        cut.push(q);
      }
    }
    return { poly, cut };
  }

  /**
   * Draws a square plot with grid and a movable decision boundary (line, tinted half-planes, normal arrow).
   */
  function makePlot(h, svg, x0, y0, size, lo, hi) {
    const id = 'clip' + (uid++);
    const sx = h.scale(lo, hi, x0, x0 + size);
    const sy = h.scale(lo, hi, y0 + size, y0);
    const defs = svg.querySelector('defs');
    const cp = h.el('clipPath', { id }, defs);
    h.el('rect', { x: x0, y: y0, width: size, height: size }, cp);
    h.el('rect', { x: x0, y: y0, width: size, height: size, rx: 6, fill: 'var(--surface-2)', stroke: 'var(--line)' }, svg);
    const g = h.el('g', { 'clip-path': `url(#${id})` }, svg);
    for (let v = Math.ceil(lo); v <= Math.floor(hi); v++) {
      const axis = v === 0;
      h.el('line', { x1: sx(v), y1: y0, x2: sx(v), y2: y0 + size, stroke: axis ? 'var(--muted)' : 'var(--line)', 'stroke-width': axis ? 1.5 : 1, opacity: axis ? 0.9 : 0.6 }, g);
      h.el('line', { x1: x0, y1: sy(v), x2: x0 + size, y2: sy(v), stroke: axis ? 'var(--muted)' : 'var(--line)', 'stroke-width': axis ? 1.5 : 1, opacity: axis ? 0.9 : 0.6 }, g);
    }
    const posP = h.el('polygon', { fill: 'var(--pos)', 'fill-opacity': 0.12 }, g);
    const negP = h.el('polygon', { fill: 'var(--neg)', 'fill-opacity': 0.12 }, g);
    const ghosts = h.el('g', {}, g);
    const line = h.el('line', { stroke: 'var(--ink)', 'stroke-width': 2.5, 'stroke-linecap': 'round', opacity: 0 }, g);
    const normal = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 3, 'marker-end': 'url(#arrow-accent)', opacity: 0 }, g);
    h.el('text', { x: x0 + size - 8, y: y0 + size - 8, 'text-anchor': 'end', 'font-size': 13, fill: 'var(--muted)', text: 'x₁' }, svg);
    h.el('text', { x: x0 + 8, y: y0 + 18, 'font-size': 13, fill: 'var(--muted)', text: 'x₂' }, svg);
    const layer = h.el('g', {}, svg);
    const over = h.el('g', { 'clip-path': `url(#${id})` }, svg);
    const toPts = (poly) => poly.map((q) => sx(q[0]).toFixed(1) + ',' + sy(q[1]).toFixed(1)).join(' ');

    /**
     * Returns the two end points (data coordinates) of the line w.x + b = 0 inside the box, or null.
     */
    function segment(w, b) {
      const cut = clipHalf(lo, hi, w, b, 1).cut;
      return cut.length >= 2 ? cut : null;
    }

    /**
     * Moves the decision boundary; showNormal also draws the weight vector from the line's foot point.
     */
    function setBoundary(w, b, showNormal) {
      const P = clipHalf(lo, hi, w, b, 1), N = clipHalf(lo, hi, w, b, -1);
      h.attr(posP, { points: toPts(P.poly) });
      h.attr(negP, { points: toPts(N.poly) });
      const c = P.cut;
      if (c.length >= 2) h.attr(line, { x1: sx(c[0][0]), y1: sy(c[0][1]), x2: sx(c[1][0]), y2: sy(c[1][1]), opacity: 1 });
      else h.attr(line, { opacity: 0 });
      const nw = Math.hypot(w[0], w[1]);
      if (showNormal && nw > 1e-6) {
        const fx = -b * w[0] / (nw * nw), fy = -b * w[1] / (nw * nw);
        const len = (hi - lo) * 0.17;
        h.attr(normal, { x1: sx(fx), y1: sy(fy), x2: sx(fx + w[0] / nw * len), y2: sy(fy + w[1] / nw * len), opacity: 1 });
      } else h.attr(normal, { opacity: 0 });
    }

    return { sx, sy, g, layer, over, ghosts, line, setBoundary, segment, x0, y0, size, lo, hi };
  }

  /**
   * Adds a data point (filled circle) to the plot's point layer.
   */
  function addPoint(h, plot, p, fill, r) {
    return h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: r || 6.5, fill, stroke: 'var(--surface)', 'stroke-width': 1.5 }, plot.layer);
  }

  /**
   * Creates a small SVG text element.
   */
  function txt(h, svg, x, y, s, extra) {
    return h.el('text', Object.assign({ x, y, text: s, 'font-size': 14, fill: 'var(--ink)' }, extra || {}), svg);
  }

  /**
   * Creates a bar chart area; returns set(values) which redraws the last bars.
   */
  function barChart(h, svg, x, y, w, hgt, maxV, title) {
    txt(h, svg, x, y - 8, title, { 'font-size': 13, fill: 'var(--muted)' });
    h.el('line', { x1: x, y1: y + hgt, x2: x + w, y2: y + hgt, stroke: 'var(--muted)', 'stroke-width': 1.5 }, svg);
    h.el('line', { x1: x, y1: y, x2: x, y2: y + hgt, stroke: 'var(--line)', 'stroke-width': 1 }, svg);
    const grp = h.el('g', {}, svg);
    const slots = 14;
    const bw = w / slots;
    return {
      set(values) {
        grp.textContent = '';
        const start = Math.max(0, values.length - slots);
        values.slice(start).forEach((v, k) => {
          const bh = (v / maxV) * (hgt - 18);
          h.el('rect', { x: x + k * bw + 3, y: y + hgt - bh, width: bw - 6, height: Math.max(bh, 0), rx: 2, fill: v === 0 ? 'var(--ok)' : 'var(--bad)', opacity: 0.85 }, grp);
          h.el('text', { x: x + k * bw + bw / 2, y: y + hgt - bh - 4, 'text-anchor': 'middle', 'font-size': 11, class: 'mono', fill: 'var(--ink)', text: String(v) }, grp);
          h.el('text', { x: x + k * bw + bw / 2, y: y + hgt + 14, 'text-anchor': 'middle', 'font-size': 10, class: 'mono', fill: 'var(--muted)', text: String(start + k + 1) }, grp);
        });
      }
    };
  }

  /**
   * Builds the drawing of a single artificial neuron (inputs, weighted edges, sum, activation, output).
   */
  function buildNeuron(h, svg, c) {
    const edges = [], wTxt = [], xVal = [];
    const x1 = c.xIn + 22, x2 = c.xSum - 26;
    c.ys.forEach((y) => {
      edges.push(h.el('line', { x1, y1: y, x2, y2: c.yC, stroke: 'var(--muted)', 'stroke-width': 2, 'stroke-linecap': 'round' }, svg));
    });
    const pos = (i, f) => ({ x: x1 + (x2 - x1) * f, y: c.ys[i] + (c.yC - c.ys[i]) * f });
    c.ys.forEach((y, i) => {
      h.el('circle', { cx: c.xIn, cy: y, r: 22, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
      xVal.push(h.el('text', { x: c.xIn, y: y + 5, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--ink)' }, svg));
      h.el('text', { x: c.xIn - 30, y: y + 5, 'text-anchor': 'end', 'font-size': 16, fill: 'var(--ink)', text: 'x' + SUB[i + 1] }, svg);
      const p = pos(i, c.wf || 0.4);
      wTxt.push(h.el('text', { x: p.x, y: p.y - 9, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', 'paint-order': 'stroke', stroke: 'var(--surface)', 'stroke-width': 4 }, svg));
    });
    h.el('line', { x1: c.xSum, y1: c.yC - 100, x2: c.xSum, y2: c.yC - 28, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
    const biasTxt = h.el('text', { x: c.xSum + 10, y: c.yC - 70, 'font-size': 13, class: 'mono' }, svg);
    h.el('circle', { cx: c.xSum, cy: c.yC, r: 26, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
    h.el('text', { x: c.xSum, y: c.yC + 8, 'text-anchor': 'middle', 'font-size': 24, fill: 'var(--ink)', text: 'Σ' }, svg);
    h.el('line', { x1: c.xSum + 26, y1: c.yC, x2: c.xAct - 32, y2: c.yC, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
    const actRect = h.el('rect', { x: c.xAct - 30, y: c.yC - 22, width: 60, height: 44, rx: 8, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
    h.el('text', { x: c.xAct, y: c.yC + 7, 'text-anchor': 'middle', 'font-size': 20, 'font-style': 'italic', fill: 'var(--ink)', text: 'g' }, svg);
    h.el('line', { x1: c.xAct + 30, y1: c.yC, x2: c.xOut - 26, y2: c.yC, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
    const outNode = h.el('circle', { cx: c.xOut, cy: c.yC, r: 24, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
    const outTxt = h.el('text', { x: c.xOut, y: c.yC + 5, 'text-anchor': 'middle', 'font-size': 14, class: 'mono', fill: 'var(--ink)' }, svg);
    const aTxt = h.el('text', { x: c.xSum, y: c.yC + 56, 'text-anchor': 'middle', 'font-size': 14, class: 'mono', fill: 'var(--ink)' }, svg);
    const gTxt = h.el('text', { x: c.xAct, y: c.yC + 42, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }, svg);
    h.el('text', { x: c.xOut, y: c.yC - 36, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'g(a)' }, svg);

    return {
      pos, edges, wTxt, xVal, aTxt, outTxt, outNode, actRect, gTxt, biasTxt,
      /**
       * Refreshes inputs, weights and bias (thickness and colour follow the sign and magnitude of each weight).
       */
      update(x, w, b) {
        w.forEach((wi, i) => {
          const col = wi > 0 ? 'var(--pos)' : wi < 0 ? 'var(--neg)' : 'var(--muted)';
          h.attr(edges[i], { 'stroke-width': 1.5 + Math.min(Math.abs(wi), 3) * 2.2, stroke: col });
          h.attr(wTxt[i], { text: `w${SUB[i + 1]} = ${num(h, wi, 1)}`, fill: col });
          h.attr(xVal[i], { text: num(h, x[i], 1) });
        });
        h.attr(biasTxt, { text: `b = ${num(h, b, 1)}`, fill: b >= 0 ? 'var(--pos)' : 'var(--neg)' });
      }
    };
  }

  Course.register({
    id: 'perceptron',
    order: 2,
    title: 'Neurone & perceptron',
    summary: 'Du neurone biologique au <b>perceptron</b> de Rosenblatt : somme pondérée, frontière de décision, algorithme d\'apprentissage… et la limite du XOR.',
    pdfPages: '11-15',
    init(shared) {
      shared.pts = makeDataset();
    },
    steps: [
      {
        title: 'Neurone biologique',
        duration: 12000,
        text: `<p>Un neurone biologique reçoit des signaux chimiques (les <b>neurotransmetteurs</b>) par ses <b>dendrites</b>.</p>
<p>Ces signaux <b>s'accumulent</b> dans le corps cellulaire (le soma). Tant que le seuil n'est pas atteint, rien ne sort.</p>
<p>Quand le <b>potentiel d'action</b> est atteint, le neurone « décharge » : un message est libéré le long de l'<b>axone</b> vers les neurones suivants.</p>`,
        note: 'Idée clé : <b>accumuler des entrées</b>, puis <b>déclencher</b> une sortie au-delà d\'un seuil. C\'est exactement ce que va imiter le neurone artificiel.',
        draw(h) {
          const svg = h.svg(720, 420);
          const paths = [
            'M60,70 Q200,80 322,188',
            'M40,170 Q170,180 316,205',
            'M50,290 Q190,270 320,228',
            'M110,372 Q230,332 335,248'
          ];
          const twigs = [[110, 72, 88, 40], [150, 77, 160, 112], [90, 176, 70, 210], [200, 187, 214, 150], [95, 283, 78, 252], [160, 275, 150, 318], [170, 346, 170, 392], [225, 338, 262, 372]];
          twigs.forEach((t) => h.el('line', { x1: t[0], y1: t[1], x2: t[2], y2: t[3], stroke: 'var(--muted)', 'stroke-width': 3, 'stroke-linecap': 'round' }, svg));
          const dend = paths.map((d) => h.el('path', { d, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 5, 'stroke-linecap': 'round' }, svg));
          const axon = h.el('path', { d: 'M406,210 L640,210', fill: 'none', stroke: 'var(--muted)', 'stroke-width': 6, 'stroke-linecap': 'round' }, svg);
          [440, 510, 580].forEach((x) => h.el('rect', { x, y: 200, width: 48, height: 20, rx: 10, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.5 }, svg));
          const terms = [[690, 176], [690, 210], [690, 244]].map((p) => {
            h.el('line', { x1: 640, y1: 210, x2: p[0], y2: p[1], stroke: 'var(--muted)', 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
            return h.el('circle', { cx: p[0], cy: p[1], r: 6, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          });
          const soma = h.el('circle', { cx: 360, cy: 210, r: 46, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
          const fill = h.el('circle', { cx: 360, cy: 210, r: 0, fill: 'var(--accent)', opacity: 0.6 }, svg);
          h.el('circle', { cx: 360, cy: 210, r: 46, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '4 4' }, svg);
          txt(h, svg, 60, 28, 'Dendrites', { 'font-size': 15 });
          txt(h, svg, 360, 142, 'Soma', { 'text-anchor': 'middle', 'font-size': 15 });
          txt(h, svg, 520, 186, 'Axone', { 'text-anchor': 'middle', 'font-size': 15 });
          txt(h, svg, 690, 280, 'Vers les autres neurones', { 'text-anchor': 'end', 'font-size': 12, fill: 'var(--muted)' });
          h.el('rect', { x: 330, y: 316, width: 130, height: 12, rx: 6, fill: 'var(--surface-2)', stroke: 'var(--line)' }, svg);
          const bar = h.el('rect', { x: 330, y: 316, width: 0, height: 12, rx: 6, fill: 'var(--accent)' }, svg);
          h.el('line', { x1: 460, y1: 310, x2: 460, y2: 334, stroke: 'var(--bad)', 'stroke-width': 2 }, svg);
          txt(h, svg, 395, 352, 'potentiel accumulé', { 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' });
          txt(h, svg, 460, 352, 'seuil', { 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--bad)' });
          const msg = txt(h, svg, 360, 400, '', { 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--muted)' });

          const pulses = [];
          const rnd = h.rng(5);
          let level = 0, nextSpawn = 300, fireT = -1;
          const dot = h.el('circle', { r: 8, fill: 'var(--accent)', opacity: 0 }, svg);

          /**
           * Launches a neurotransmitter pulse on dendrite k carrying amt units of potential.
           */
          function spawn(k, amt) {
            const c = h.el('circle', { r: 6.5, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 1.5 }, svg);
            pulses.push({ k, c, t0: null, amt, len: dend[k].getTotalLength() });
          }

          h.button('Stimulation forte', () => { for (let k = 0; k < 4; k++) spawn(k, 0.3); }, 'primary');

          h.loop((t, dt) => {
            if (t >= nextSpawn) {
              spawn(Math.floor(rnd() * 4), 0.16);
              nextSpawn = t + 420 + rnd() * 500;
            }
            for (let i = pulses.length - 1; i >= 0; i--) {
              const p = pulses[i];
              if (p.t0 === null) p.t0 = t;
              const q = (t - p.t0) / 1100;
              if (q >= 1) {
                if (fireT < 0) level += p.amt;
                p.c.remove();
                pulses.splice(i, 1);
              } else {
                const pt = dend[p.k].getPointAtLength(q * p.len);
                h.attr(p.c, { cx: pt.x, cy: pt.y });
              }
            }
            if (fireT < 0) level = Math.max(0, level - dt * 0.00005);
            if (level >= 1 && fireT < 0) { fireT = t; level = 1; }
            let strokeCol = 'var(--ink)', sw = 2;
            if (fireT >= 0) {
              const q = (t - fireT) / 1500;
              strokeCol = 'var(--accent)'; sw = 5;
              level = Math.max(0, 1 - q / 0.3);
              const a = Math.min(q / 0.7, 1);
              h.attr(dot, { cx: 406 + 234 * a, cy: 210, opacity: q < 0.72 ? 1 : 0 });
              terms.forEach((c) => h.attr(c, { fill: q >= 0.7 && q < 0.95 ? 'var(--accent)' : 'var(--surface)' }));
              h.attr(axon, { stroke: 'var(--accent)' });
              h.attr(msg, { text: 'Seuil atteint : le message part le long de l\'axone !', fill: 'var(--accent)' });
              if (q >= 1) {
                fireT = -1;
                level = 0;
                h.attr(dot, { opacity: 0 });
                h.attr(axon, { stroke: 'var(--muted)' });
                h.attr(msg, { text: '' });
              }
            } else {
              h.attr(msg, { text: level > 0.05 ? 'Les signaux s\'accumulent…' : '', fill: 'var(--muted)' });
            }
            const lv = Math.min(level, 1);
            h.attr(fill, { r: 44 * Math.sqrt(lv) });
            h.attr(soma, { stroke: strokeCol, 'stroke-width': sw });
            h.attr(bar, { width: 130 * lv });
          });
        }
      },
      {
        title: 'Neurone artificiel',
        duration: 15000,
        text: `<p>Le neurone artificiel reprend cette idée. Il reçoit $d$ entrées $x_1,\\dots,x_d$, chacune pondérée par un <b>poids</b> $w_i$ (l'importance de la « dendrite »).</p>
<p>Il <b>accumule</b> d'abord une somme pondérée plus un biais $b$ (le niveau de repos) : la <b>pré-activation</b>.</p>
$$a=\\sum_{i=1}^{d} w_i x_i + b = \\mathbf{w}^\\top\\mathbf{x}+b$$
<p>Puis une fonction d'<b>activation</b> $g$ décide de la sortie : $\\hat y = g(a)$ (le « seuil » du potentiel d'action).</p>`,
        note: 'Un poids positif <b>excite</b>, un poids négatif <b>inhibe</b>. Le biais $b$ décale le seuil de déclenchement.',
        draw(h) {
          const svg = h.svg(720, 420);
          const N = buildNeuron(h, svg, { ys: [90, 210, 330], yC: 210, xIn: 110, xSum: 380, xAct: 510, xOut: 630, wf: 0.4 });
          const x = [1.0, 0.5, -1.0], w = [0.8, 1.0, 0.6], b = -0.2;
          const prods = x.map((xi, i) => xi * w[i]);
          N.update(x, w, b);
          h.attr(N.gTxt, { text: 'échelon / signe' });
          const prodTxt = prods.map((p, i) => {
            const q = N.pos(i, 0.74);
            return h.el('text', { x: q.x, y: q.y + 22, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--accent)', opacity: 0, 'paint-order': 'stroke', stroke: 'var(--surface)', 'stroke-width': 4, text: num(h, p, 2) }, svg);
          });
          const cap = txt(h, svg, 360, 404, '', { 'text-anchor': 'middle', 'font-size': 14 });
          h.attr(N.biasTxt, { opacity: 1 });

          /**
           * Moves an accent token from p1 to p2.
           */
          async function token(p1, p2, ms) {
            const c = h.el('circle', { cx: p1.x, cy: p1.y, r: 7, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 1.5 }, svg);
            await h.tween(ms, (q) => h.attr(c, { cx: h.lerp(p1.x, p2.x, q), cy: h.lerp(p1.y, p2.y, q) }));
            c.remove();
          }

          (async () => {
            while (h.alive()) {
              prodTxt.forEach((t) => h.attr(t, { opacity: 0 }));
              h.attr(N.aTxt, { text: '' });
              h.attr(N.outTxt, { text: '' });
              h.attr(N.outNode, { fill: 'var(--surface)' });
              h.attr(N.actRect, { fill: 'var(--surface)' });
              h.attr(cap, { text: '① Chaque entrée est multipliée par son poids : wᵢ · xᵢ' });
              await h.sleep(600);
              let run = 0;
              for (let i = 0; i < 3; i++) {
                await token(N.pos(i, 0), N.pos(i, 1), 700);
                h.attr(prodTxt[i], { opacity: 1 });
                run += prods[i];
                h.attr(N.aTxt, { text: `Σ wᵢxᵢ = ${num(h, run, 2)}` });
              }
              h.attr(cap, { text: '② On ajoute le biais b' });
              await token({ x: 380, y: 110 }, { x: 380, y: 184 }, 600);
              run += b;
              h.attr(N.aTxt, { text: `a = ${num(h, run, 2)}` });
              await h.sleep(500);
              h.attr(cap, { text: '③ La fonction d\'activation g décide de la sortie' });
              await token({ x: 406, y: 210 }, { x: 476, y: 210 }, 500);
              h.attr(N.actRect, { fill: 'var(--accent)', 'fill-opacity': 0.35 });
              await token({ x: 540, y: 210 }, { x: 604, y: 210 }, 500);
              h.attr(N.outTxt, { text: run >= 0 ? '+1' : '−1' });
              h.attr(N.outNode, { fill: run >= 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.35 });
              await h.sleep(2400);
            }
          })();
        }
      },
      {
        title: 'Jouer avec le neurone',
        duration: 14000,
        text: `<p>À vous : modifiez les entrées, les poids et le biais, et observez la sortie en direct.</p>
<p>Deux activations classiques :</p>
<ul><li><b>échelon</b> : $g(a)=1$ si $a\\ge 0$, sinon $0$ ;</li><li><b>signe</b> : $g(a)=+1$ si $a\\ge 0$, sinon $-1$.</li></ul>
<p>Le perceptron utilise le signe : sa sortie est une <b>classe</b> $\\hat y\\in\\{-1,+1\\}$.</p>`,
        note: 'Le neurone ne fait que : produit scalaire $\\mathbf{w}^\\top\\mathbf{x}$, plus un biais, puis un seuil. C\'est une fonction <b>linéaire</b> suivie d\'une décision.',
        check: {
          q: 'Pour $x=(1,2)$, $w=(2,-1)$, $b=-1$ et $g$ = signe, quelle est la sortie ?',
          choices: ['$+1$', '$-1$', '$0$'],
          answer: 1,
          explain: '$a = 2\\cdot1 + (-1)\\cdot2 - 1 = -1 < 0$, donc la sortie vaut $-1$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const N = buildNeuron(h, svg, { ys: [130, 290], yC: 210, xIn: 90, xSum: 270, xAct: 360, xOut: 450, wf: 0.45 });
          const px = 520, py = 130, pw = 170, ph = 170;
          const amax = 6;
          const sa = h.scale(-amax, amax, px, px + pw);
          const sg = h.scale(-1.5, 1.5, py + ph, py);
          h.el('rect', { x: px, y: py, width: pw, height: ph, rx: 6, fill: 'var(--surface-2)', stroke: 'var(--line)' }, svg);
          h.el('line', { x1: px, y1: sg(0), x2: px + pw, y2: sg(0), stroke: 'var(--muted)', 'stroke-width': 1 }, svg);
          h.el('line', { x1: sa(0), y1: py, x2: sa(0), y2: py + ph, stroke: 'var(--muted)', 'stroke-width': 1 }, svg);
          txt(h, svg, px + pw / 2, py - 12, 'Activation g(a)', { 'text-anchor': 'middle', 'font-size': 14 });
          txt(h, svg, px + pw - 4, py + ph + 18, 'a', { 'text-anchor': 'end', 'font-size': 13, fill: 'var(--muted)' });
          const curve = h.el('path', { fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2.5 }, svg);
          const mark = h.el('circle', { r: 7, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
          const guide = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
          let mode = 'sign';
          const st = { x: [1.0, -0.5], w: [1.2, 0.8], b: -0.5 };

          /**
           * Activation function for the current mode.
           */
          const g = (a) => (a >= 0 ? 1 : (mode === 'sign' ? -1 : 0));

          /**
           * Redraws the whole neuron for the current slider state.
           */
          function refresh() {
            N.update(st.x, st.w, st.b);
            const a = st.x[0] * st.w[0] + st.x[1] * st.w[1] + st.b;
            const y = g(a);
            const low = mode === 'sign' ? -1 : 0;
            h.attr(N.aTxt, { text: `a = ${num(h, a, 2)}` });
            h.attr(N.outTxt, { text: num(h, y, 0) });
            h.attr(N.outNode, { fill: y > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.35 });
            h.attr(N.gTxt, { text: mode === 'sign' ? 'signe' : 'échelon' });
            h.attr(curve, { d: `M${px},${sg(low)} L${sa(0)},${sg(low)} L${sa(0)},${sg(1)} L${px + pw},${sg(1)}` });
            const ac = Math.max(-amax, Math.min(amax, a));
            h.attr(mark, { cx: sa(ac), cy: sg(y) });
            h.attr(guide, { x1: sa(ac), y1: sg(0), x2: sa(ac), y2: sg(y) });
          }

          const mk = (label, key, arr, idx, lo, hi, val) => h.slider({
            label, min: lo, max: hi, step: 0.1, value: val, decimals: 1,
            onInput: (v) => { if (arr) st[key][idx] = v; else st[key] = v; refresh(); }
          });
          mk('$x_1$', 'x', true, 0, -2, 2, st.x[0]);
          mk('$x_2$', 'x', true, 1, -2, 2, st.x[1]);
          mk('$w_1$', 'w', true, 0, -2, 2, st.w[0]);
          mk('$w_2$', 'w', true, 1, -2, 2, st.w[1]);
          mk('$b$', 'b', false, 0, -3, 3, st.b);
          h.toggle([['sign', 'Signe'], ['step', 'Échelon']], 'sign', (m) => { mode = m; refresh(); });
          refresh();
        }
      },
      {
        title: 'Un classifieur linéaire',
        duration: 14000,
        text: `<p>En dimension 2, le perceptron classe un point $\\mathbf{x}=(x_1,x_2)$ selon le <b>signe</b> de $\\mathbf{w}^\\top\\mathbf{x}+b$.</p>
<p>La frontière de décision est l'ensemble des points où $\\mathbf{w}^\\top\\mathbf{x}+b=0$ : une <b>droite</b>.</p>
<p>Le vecteur $\\mathbf{w}$ est <b>normal</b> à cette droite et pointe vers le côté « $+1$ ». Le biais $b$ la <b>déplace</b> sans la tourner.</p>`,
        note: 'En dimension $d$, la frontière est un <b>hyperplan</b>. Un perceptron ne sait séparer que des données <b>linéairement séparables</b>.',
        check: {
          q: 'Quelle est la position du vecteur $\\mathbf{w}$ par rapport à la droite $\\mathbf{w}^\\top\\mathbf{x}+b=0$ ?',
          choices: ['Parallèle à la droite', 'Orthogonal (normal) à la droite', 'Aucun lien géométrique'],
          answer: 1,
          explain: 'Deux points $u,v$ de la droite vérifient $\\mathbf{w}^\\top(u-v)=0$ : $\\mathbf{w}$ est orthogonal à la direction de la droite.'
        },
        draw(h) {
          const pts = h.shared.pts, n = pts.length;
          const svg = h.svg(720, 420);
          const plot = makePlot(h, svg, 20, 20, 380, -4, 4);
          const rings = [], dots = [];
          pts.forEach((p) => {
            rings.push(h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: 11, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 2.5, opacity: 0 }, plot.layer));
            dots.push(addPoint(h, plot, p, 'var(--muted)'));
          });
          h.label(svg, 430, 14, 280, 40, '$\\hat y=\\mathrm{signe}(w_1x_1+w_2x_2+b)$');
          const tW = txt(h, svg, 430, 90, '', { class: 'mono' });
          const tB = txt(h, svg, 430, 114, '', { class: 'mono' });
          const tN = txt(h, svg, 430, 138, '', { class: 'mono' });
          const tD = txt(h, svg, 430, 162, '', { class: 'mono' });
          const tE = txt(h, svg, 430, 204, '', { class: 'mono', 'font-size': 15 });
          h.el('circle', { cx: 438, cy: 256, r: 6.5, fill: 'var(--pos)' }, svg);
          txt(h, svg, 452, 261, 'prédit +1 (zone verte)', { 'font-size': 13 });
          h.el('circle', { cx: 438, cy: 282, r: 6.5, fill: 'var(--neg)' }, svg);
          txt(h, svg, 452, 287, 'prédit −1 (zone rose)', { 'font-size': 13 });
          h.el('circle', { cx: 438, cy: 308, r: 9, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 2.5 }, svg);
          txt(h, svg, 452, 313, 'point mal classé', { 'font-size': 13 });
          h.el('line', { x1: 430, y1: 340, x2: 446, y2: 340, stroke: 'var(--accent)', 'stroke-width': 3, 'marker-end': 'url(#arrow-accent)' }, svg);
          txt(h, svg, 456, 345, 'vecteur w (normal)', { 'font-size': 13 });
          const st = { w: [0.4, 1.4], b: 1.2 };

          /**
           * Redraws the boundary, the point colours and the numeric read-outs.
           */
          function refresh() {
            plot.setBoundary(st.w, st.b, true);
            let bad = 0;
            pts.forEach((p, i) => {
              const pred = score(st.w, st.b, p) >= 0 ? 1 : -1;
              h.attr(dots[i], { fill: colorOf(pred) });
              const wrong = pred !== p.c;
              if (wrong) bad++;
              h.attr(rings[i], { opacity: wrong ? 1 : 0 });
            });
            const nw = Math.hypot(st.w[0], st.w[1]);
            h.attr(tW, { text: `w = (${num(h, st.w[0], 2)} ; ${num(h, st.w[1], 2)})` });
            h.attr(tB, { text: `b = ${num(h, st.b, 2)}` });
            h.attr(tN, { text: `‖w‖ = ${num(h, nw, 2)}` });
            h.attr(tD, { text: `dist. à l'origine = ${num(h, nw > 1e-6 ? Math.abs(st.b) / nw : 0, 2)}` });
            h.attr(tE, { text: `Mal classés : ${bad} / ${n}`, fill: bad === 0 ? 'var(--ok)' : 'var(--ink)' });
          }

          h.slider({ label: '$w_1$', min: -3, max: 3, step: 0.05, value: st.w[0], onInput: (v) => { st.w[0] = v; refresh(); } });
          h.slider({ label: '$w_2$', min: -3, max: 3, step: 0.05, value: st.w[1], onInput: (v) => { st.w[1] = v; refresh(); } });
          h.slider({ label: '$b$', min: -4, max: 4, step: 0.05, value: st.b, onInput: (v) => { st.b = v; refresh(); } });
          refresh();
        }
      },
      {
        title: 'Corriger une erreur',
        duration: 15000,
        text: `<p>Comment trouver $\\mathbf{w}$ et $b$ automatiquement ? <b>Rosenblatt (1957)</b> propose une règle très simple. Avec des étiquettes $y\\in\\{-1,+1\\}$ :</p>
<p>on parcourt les exemples, et si $(\\mathbf{x},y)$ est <b>mal classé</b> ($y\\,(\\mathbf{w}^\\top\\mathbf{x}+b)\\le 0$), on corrige :</p>
$$\\mathbf{w}\\leftarrow \\mathbf{w}+\\eta\\, y\\,\\mathbf{x},\\qquad b\\leftarrow b+\\eta\\, y$$
<p>Ici $y=+1$ : on <b>ajoute</b> $\\eta\\mathbf{x}$ à $\\mathbf{w}$, qui se rapproche de $\\mathbf{x}$.</p>`,
        note: '$\\eta>0$ est le <b>pas d\'apprentissage</b>. Si le point est du bon côté, on ne change rien.',
        check: {
          q: 'Un point d\'étiquette $y=-1$ est classé $+1$. Que fait la règle de mise à jour ?',
          choices: ['$\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta\\mathbf{x}$', '$\\mathbf{w}\\leftarrow\\mathbf{w}-\\eta\\mathbf{x}$', '$\\mathbf{w}$ reste inchangé'],
          answer: 1,
          explain: 'Avec $y=-1$, $\\eta y \\mathbf{x}=-\\eta\\mathbf{x}$ : $\\mathbf{w}$ s\'éloigne de $\\mathbf{x}$, ce qui diminue $\\mathbf{w}^\\top\\mathbf{x}$ et pousse le point vers la classe $-1$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const plot = makePlot(h, svg, 20, 20, 380, -4, 4);
          const ctx = [
            { x: -1.5, y: 2, c: 1 }, { x: -0.5, y: 1.8, c: 1 }, { x: -3, y: 2.5, c: 1 },
            { x: 1, y: -2, c: -1 }, { x: 2, y: -2.5, c: -1 }, { x: 0.5, y: -2.2, c: -1 }
          ];
          const focus = { x: 2, y: 1, c: 1 };
          ctx.forEach((p) => addPoint(h, plot, p, colorOf(p.c)));
          const ring = h.el('circle', { cx: plot.sx(focus.x), cy: plot.sy(focus.y), r: 15, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3, opacity: 0 }, plot.layer);
          const fdot = addPoint(h, plot, focus, 'var(--neg)', 8);
          const O = [plot.sx(0), plot.sy(0)];
          const wArrow = h.el('line', { stroke: 'var(--ink)', 'stroke-width': 3, 'marker-end': 'url(#arrow)' }, plot.over);
          const dArrow = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 3, 'stroke-dasharray': '6 4', 'marker-end': 'url(#arrow-accent)', opacity: 0 }, plot.over);
          h.el('circle', { cx: O[0], cy: O[1], r: 3.5, fill: 'var(--ink)' }, plot.over);
          const wLab = h.el('text', { 'font-size': 15, 'font-weight': 'bold', fill: 'var(--ink)', text: 'w' }, plot.over);
          txt(h, svg, 430, 32, 'Point courant', { 'font-size': 15, 'font-weight': 'bold' });
          txt(h, svg, 430, 56, 'x = (2,00 ; 1,00), y = +1', { class: 'mono' });
          const tA = txt(h, svg, 430, 90, '', { class: 'mono' });
          const tC = txt(h, svg, 430, 116, '', { class: 'mono' });
          const tW = txt(h, svg, 430, 160, '', { class: 'mono' });
          const tB = txt(h, svg, 430, 184, '', { class: 'mono' });
          const tM = txt(h, svg, 430, 230, '', { 'font-size': 15 });
          h.label(svg, 430, 262, 280, 40, '$\\mathbf{w}\\leftarrow\\mathbf{w}+\\eta\\,y\\,\\mathbf{x}$');
          h.label(svg, 430, 300, 280, 40, '$b\\leftarrow b+\\eta\\,y$');
          const w0 = [-1, 0.3], b0 = 0;
          let eta = 1, token = 0;

          /**
           * Shows the model (w arrow, line, point colour) for weights w and bias b.
           */
          function render(w, b) {
            plot.setBoundary(w, b, false);
            const tip = [plot.sx(w[0]), plot.sy(w[1])];
            h.attr(wArrow, { x1: O[0], y1: O[1], x2: tip[0], y2: tip[1] });
            h.attr(wLab, { x: tip[0] + (w[0] >= 0 ? 8 : -18), y: tip[1] - 6 });
            const a = score(w, b, focus);
            h.attr(tA, { text: `a = w·x + b = ${num(h, a, 2)}` });
            h.attr(tC, { text: `classé ${a >= 0 ? '+1' : '−1'}`, fill: a >= 0 ? 'var(--ok)' : 'var(--bad)' });
            h.attr(fdot, { fill: colorOf(a >= 0 ? 1 : -1) });
            h.attr(tW, { text: `w = (${num(h, w[0], 2)} ; ${num(h, w[1], 2)})` });
            h.attr(tB, { text: `b = ${num(h, b, 2)}` });
          }

          /**
           * Replays the correction animation from the initial model with the current eta.
           */
          async function run() {
            const my = ++token;
            h.attr(dArrow, { opacity: 0 });
            h.attr(ring, { opacity: 0 });
            h.attr(tM, { text: '' });
            render(w0, b0);
            await h.sleep(900);
            if (my !== token) return;
            h.attr(ring, { opacity: 1 });
            h.attr(tM, { text: 'Erreur : le point +1 est classé −1', fill: 'var(--bad)' });
            await h.sleep(1500);
            if (my !== token) return;
            const tip = [plot.sx(w0[0]), plot.sy(w0[1])];
            const end = [plot.sx(w0[0] + eta * focus.x), plot.sy(w0[1] + eta * focus.y)];
            h.attr(dArrow, { x1: tip[0], y1: tip[1], x2: tip[0], y2: tip[1], opacity: 1 });
            h.attr(tM, { text: 'On ajoute η·y·x à w', fill: 'var(--accent)' });
            await h.tween(900, (q) => { if (my === token) h.attr(dArrow, { x2: h.lerp(tip[0], end[0], q), y2: h.lerp(tip[1], end[1], q) }); });
            if (my !== token) return;
            await h.sleep(500);
            if (my !== token) return;
            await h.tween(1400, (q) => {
              if (my !== token) return;
              render([w0[0] + q * eta * focus.x, w0[1] + q * eta * focus.y], b0 + q * eta);
            });
            if (my !== token) return;
            h.attr(tM, { text: 'La droite a pivoté : le point est bien classé', fill: 'var(--ok)' });
            h.attr(ring, { opacity: 0 });
          }

          h.slider({ label: '$\\eta$', min: 0.2, max: 1.5, step: 0.1, value: 1, decimals: 1, onInput: (v) => { eta = v; run(); } });
          h.button('Rejouer la correction', () => run(), 'primary');
          run();
        }
      },
      {
        title: 'Entraîner pas à pas',
        duration: 20000,
        text: `<p>Voici l'algorithme complet. On répète sur des <b>époques</b> (un passage sur tous les exemples) :</p>
<ol><li>prendre l'exemple suivant $(\\mathbf{x}_i,y_i)$ ;</li><li>si $y_i(\\mathbf{w}^\\top\\mathbf{x}_i+b)\\le0$ : mettre à jour $\\mathbf{w}$ et $b$ ;</li><li>sinon : ne rien faire.</li></ol>
<p>Utilisez les boutons : une itération, une époque, ou l'entraînement automatique. La flèche en pointillés est la correction $\\eta\\,y\\,\\mathbf{x}$ ajoutée à $\\mathbf{w}$.</p>`,
        note: 'On s\'arrête quand une époque entière se passe <b>sans aucune erreur</b> : les données sont alors parfaitement séparées.',
        draw(h) {
          const pts = h.shared.pts, n = pts.length;
          const svg = h.svg(720, 420);
          const plot = makePlot(h, svg, 20, 20, 380, -4, 4);
          const rings = [], dots = [];
          pts.forEach((p) => {
            rings.push(h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: 11, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 2.5, opacity: 0 }, plot.layer));
            dots.push(addPoint(h, plot, p, colorOf(p.c)));
          });
          const cur = h.el('circle', { r: 15, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3, opacity: 0 }, plot.layer);
          const O = [plot.sx(0), plot.sy(0)];
          const wArrow = h.el('line', { stroke: 'var(--ink)', 'stroke-width': 2.5, 'marker-end': 'url(#arrow)' }, plot.over);
          const dArrow = h.el('line', { stroke: 'var(--accent)', 'stroke-width': 3, 'stroke-dasharray': '6 4', 'marker-end': 'url(#arrow-accent)', opacity: 0 }, plot.over);
          h.el('circle', { cx: O[0], cy: O[1], r: 3.5, fill: 'var(--ink)' }, plot.over);
          const tE = txt(h, svg, 430, 34, '', { class: 'mono' });
          const tI = txt(h, svg, 430, 58, '', { class: 'mono' });
          const tR = txt(h, svg, 430, 82, '', { class: 'mono' });
          const tM = txt(h, svg, 430, 106, '', { class: 'mono' });
          const tW = txt(h, svg, 430, 140, '', { class: 'mono' });
          const tD = txt(h, svg, 430, 164, '', { class: 'mono', fill: 'var(--accent)' });
          const tS = txt(h, svg, 430, 202, '', { 'font-size': 15 });
          const chart = barChart(h, svg, 436, 250, 260, 120, n, 'Erreurs par époque');
          const W0 = [0.5, -0.9], B0 = 0.5;
          const st = { w: W0.slice(), b: B0, i: 0, epoch: 0, errs: 0, hist: [], gen: 0, busy: false, running: false, done: false, msg: '', delta: '' };

          /**
           * Draws the model: boundary, w arrow and error rings.
           */
          function render(w, b) {
            plot.setBoundary(w, b, false);
            h.attr(wArrow, { x1: O[0], y1: O[1], x2: plot.sx(w[0]), y2: plot.sy(w[1]) });
            pts.forEach((p, i) => h.attr(rings[i], { opacity: isErr(w, b, p) ? 1 : 0 }));
          }

          /**
           * Refreshes the text read-outs from the training state.
           */
          function info() {
            h.attr(tE, { text: `Époque : ${st.epoch}` });
            h.attr(tI, { text: `Prochain point : ${st.i + 1} / ${n}` });
            h.attr(tR, { text: `Corrections (époque) : ${st.errs}` });
            h.attr(tM, { text: `Mal classés (modèle) : ${countErr(st.w, st.b, pts)} / ${n}` });
            h.attr(tW, { text: `w=(${num(h, st.w[0], 2)} ; ${num(h, st.w[1], 2)}) b=${num(h, st.b, 2)}` });
            h.attr(tD, { text: st.delta });
            h.attr(tS, { text: st.msg, fill: st.done ? 'var(--ok)' : 'var(--ink)' });
          }

          const etaS = h.slider({ label: '$\\eta$', min: 0.1, max: 1, step: 0.1, value: 0.5, decimals: 1, onInput: () => {} });

          /**
           * Processes the current example; fast shortens the animation.
           */
          async function processOne(fast) {
            const g = st.gen;
            const p = pts[st.i];
            h.attr(cur, { cx: plot.sx(p.x), cy: plot.sy(p.y), opacity: 1 });
            h.attr(dArrow, { opacity: 0 });
            st.delta = '';
            st.msg = `Point ${st.i + 1} : y = ${p.c > 0 ? '+1' : '−1'}`;
            info();
            await h.sleep(fast ? 70 : 500);
            if (g !== st.gen) return;
            const a = score(st.w, st.b, p);
            if (p.c * a <= 0) {
              st.errs++;
              const eta = etaS.get();
              const dw = [eta * p.c * p.x, eta * p.c * p.y], db = eta * p.c;
              const w0 = st.w.slice(), b0 = st.b;
              const tip = [plot.sx(w0[0]), plot.sy(w0[1])];
              h.attr(dArrow, { x1: tip[0], y1: tip[1], x2: plot.sx(w0[0] + dw[0]), y2: plot.sy(w0[1] + dw[1]), opacity: 1 });
              st.delta = `ηyx = (${num(h, dw[0], 2)} ; ${num(h, dw[1], 2)})`;
              st.msg = 'Erreur : on corrige w et b';
              info();
              await h.sleep(fast ? 90 : 700);
              if (g !== st.gen) return;
              await h.tween(fast ? 250 : 800, (q) => {
                if (g !== st.gen) return;
                render([w0[0] + dw[0] * q, w0[1] + dw[1] * q], b0 + db * q);
              });
              if (g !== st.gen) return;
              st.w = [w0[0] + dw[0], w0[1] + dw[1]];
              st.b = b0 + db;
              h.attr(dArrow, { opacity: 0 });
            } else {
              st.msg = 'Bien classé : rien à faire';
              if (!fast) await h.sleep(300);
              if (g !== st.gen) return;
            }
            st.i++;
            if (st.i >= n) {
              st.hist.push(st.errs);
              chart.set(st.hist);
              st.epoch++;
              if (st.errs === 0) {
                st.done = true;
                st.msg = `Convergé en ${st.epoch} époques : 0 erreur !`;
              } else st.msg = `Fin de l'époque ${st.epoch} : ${st.errs} correction(s)`;
              st.errs = 0;
              st.i = 0;
              h.attr(cur, { opacity: 0 });
            }
            info();
          }

          const btnTrain = h.button('Entraîner', () => autoTrain(), 'primary');

          /**
           * Resets weights, counters and visuals.
           */
          function reset() {
            st.gen++;
            Object.assign(st, { w: W0.slice(), b: B0, i: 0, epoch: 0, errs: 0, hist: [], busy: false, running: false, done: false, msg: 'Prêt : modèle initial', delta: '' });
            h.attr(cur, { opacity: 0 });
            h.attr(dArrow, { opacity: 0 });
            btnTrain.innerHTML = 'Entraîner';
            render(st.w, st.b);
            chart.set([]);
            info();
          }

          /**
           * Runs one example when idle.
           */
          async function oneStep() {
            if (st.busy || st.done) return;
            const g = st.gen;
            st.busy = true;
            await processOne(false);
            if (g === st.gen) st.busy = false;
          }

          /**
           * Runs until the end of the current epoch.
           */
          async function oneEpoch() {
            if (st.busy || st.done) return;
            const g = st.gen;
            st.busy = true;
            const target = st.epoch + 1;
            while (g === st.gen && st.epoch < target && !st.done) await processOne(true);
            if (g === st.gen) st.busy = false;
          }

          /**
           * Toggles automatic training until convergence.
           */
          async function autoTrain() {
            if (st.running) { st.running = false; btnTrain.innerHTML = 'Entraîner'; return; }
            if (st.busy || st.done) return;
            const g = st.gen;
            st.busy = true;
            st.running = true;
            btnTrain.innerHTML = 'Pause';
            while (g === st.gen && st.running && !st.done && st.epoch < 40) await processOne(true);
            if (g === st.gen) { st.busy = false; st.running = false; btnTrain.innerHTML = 'Entraîner'; }
          }

          h.button('Une itération', oneStep);
          h.button('Une époque', oneEpoch);
          h.button('Réinitialiser', reset);
          reset();
        }
      },
      {
        title: 'Convergence',
        duration: 18000,
        text: `<p>Si les données sont <b>linéairement séparables</b>, l'algorithme de Rosenblatt <b>converge</b> toujours en un nombre fini de corrections (théorème de Novikoff : au plus $(R/\\gamma)^2$, avec $R$ le rayon des données et $\\gamma$ la marge).</p>
<p>Sinon, il y aura toujours des erreurs : la droite oscille indéfiniment.</p>
<p>Comparez les deux cas avec le sélecteur (dans le second, deux étiquettes ont été inversées).</p>`,
        note: 'Plus la marge $\\gamma$ est grande, plus la convergence est rapide. Sans séparabilité, aucune garantie : il faut un autre modèle.',
        check: {
          q: 'Que se passe-t-il si les données ne sont pas linéairement séparables ?',
          choices: ['L\'algorithme converge plus lentement', 'Il ne converge jamais : il reste des erreurs à chaque époque', 'Il converge vers la meilleure droite'],
          answer: 1,
          explain: 'Aucune droite ne classe tout correctement : chaque époque contient au moins une erreur, donc des mises à jour sans fin.'
        },
        draw(h) {
          const sep = h.shared.pts;
          const non = makeNoisy(sep);
          const svg = h.svg(720, 420);
          const plot = makePlot(h, svg, 20, 20, 380, -4, 4);
          const ptsG = h.el('g', {}, plot.layer);
          const tD = txt(h, svg, 430, 34, '', { 'font-size': 15, 'font-weight': 'bold' });
          const tE = txt(h, svg, 430, 62, '', { class: 'mono' });
          const tM1 = txt(h, svg, 430, 94, '', { 'font-size': 14 });
          const tM2 = txt(h, svg, 430, 114, '', { 'font-size': 14 });
          const chart = barChart(h, svg, 436, 170, 260, 200, sep.length, 'Corrections par époque');
          let session = 0;

          /**
           * Rebuilds the point markers for a dataset and returns their ring elements.
           */
          function buildPoints(data) {
            ptsG.textContent = '';
            const rings = data.map((p) => h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: 11, fill: 'none', stroke: 'var(--bad)', 'stroke-width': 2.5, opacity: 0 }, ptsG));
            data.forEach((p) => h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: 6.5, fill: colorOf(p.c), stroke: 'var(--surface)', 'stroke-width': 1.5 }, ptsG));
            return rings;
          }

          /**
           * Plays the training trace of one dataset mode ('sep' or 'non').
           */
          async function run(mode) {
            const my = ++session;
            const data = mode === 'sep' ? sep : non;
            const rings = buildPoints(data);
            const snaps = trainTrace(data, 0.5, mode === 'sep' ? 30 : 14);
            const show = (w, b) => {
              plot.setBoundary(w, b, false);
              data.forEach((p, i) => h.attr(rings[i], { opacity: isErr(w, b, p) ? 1 : 0 }));
            };
            chart.set([]);
            h.attr(tD, { text: mode === 'sep' ? 'Données séparables' : 'Données non séparables' });
            h.attr(tE, { text: 'Époque 0 (modèle initial)' });
            h.attr(tM1, { text: '' });
            h.attr(tM2, { text: '' });
            show(snaps[0].w, snaps[0].b);
            await h.sleep(900);
            const vals = [];
            for (let k = 1; k < snaps.length; k++) {
              if (my !== session) return;
              const a = snaps[k - 1], c = snaps[k];
              h.attr(tE, { text: `Époque ${k} en cours…` });
              await h.tween(600, (q) => {
                if (my !== session) return;
                show([h.lerp(a.w[0], c.w[0], q), h.lerp(a.w[1], c.w[1], q)], h.lerp(a.b, c.b, q));
              });
              if (my !== session) return;
              vals.push(c.err);
              chart.set(vals);
              h.attr(tE, { text: `Époque ${k} : ${c.err} correction(s)` });
              await h.sleep(300);
            }
            if (my !== session) return;
            if (mode === 'sep') {
              h.attr(tM1, { text: `0 erreur à l'époque ${snaps.length - 1} :`, fill: 'var(--ok)' });
              h.attr(tM2, { text: 'convergence garantie.', fill: 'var(--ok)' });
            } else {
              h.attr(tM1, { text: 'Toujours des erreurs : la droite', fill: 'var(--bad)' });
              h.attr(tM2, { text: 'oscille sans jamais converger.', fill: 'var(--bad)' });
            }
          }

          h.toggle([['sep', 'Séparable'], ['non', 'Non séparable']], 'sep', (m) => run(m));
          run('sep');
        }
      },
      {
        title: 'Le problème du XOR',
        duration: 16000,
        text: `<p>En 1969, <b>Minsky et Papert</b> montrent une limite fondamentale du perceptron : il ne résout pas le <b>XOR</b> (« ou exclusif »).</p>
<p>Le XOR vaut $+1$ si les deux entrées sont <b>différentes</b>, $-1$ sinon. Les points $(0,1)$ et $(1,0)$ sont d'une classe, $(0,0)$ et $(1,1)$ de l'autre.</p>
<p>Regardez la droite tourner : à chaque angle, on prend le meilleur décalage possible. Il reste <b>toujours au moins une erreur</b>.</p>`,
        note: 'Aucune droite ne sépare les diagonales d\'un carré : le XOR n\'est <b>pas linéairement séparable</b>. Ce constat a freiné la recherche sur les réseaux de neurones pendant des années.',
        check: {
          q: 'Quel est le meilleur score possible d\'un perceptron sur les 4 points du XOR ?',
          choices: ['4 sur 4', '3 sur 4', '2 sur 4'],
          answer: 1,
          explain: 'On peut classer correctement 3 points, mais jamais les 4 : une droite ne peut pas isoler les deux points d\'une même diagonale.'
        },
        draw(h) {
          const X = [
            { x: 0, y: 0, c: -1 }, { x: 1, y: 1, c: -1 }, { x: 0, y: 1, c: 1 }, { x: 1, y: 0, c: 1 }
          ];
          const svg = h.svg(720, 420);
          const plot = makePlot(h, svg, 20, 20, 380, -0.5, 1.5);
          X.forEach((p) => {
            h.el('circle', { cx: plot.sx(p.x), cy: plot.sy(p.y), r: 11, fill: colorOf(p.c), stroke: 'var(--surface)', 'stroke-width': 2 }, plot.layer);
            const dx = p.x === 1 ? -14 : 14;
            h.el('text', { x: plot.sx(p.x) + dx, y: plot.sy(p.y) + (p.y === 1 ? 26 : -18), 'text-anchor': p.x === 1 ? 'end' : 'start', 'font-size': 13, class: 'mono', fill: 'var(--ink)', text: `(${p.x},${p.y}) → ${p.c > 0 ? '+1' : '−1'}` }, plot.layer);
          });
          const tT = txt(h, svg, 430, 40, '', { 'font-size': 15, 'font-weight': 'bold' });
          const tA = txt(h, svg, 430, 72, '', { class: 'mono' });
          const tE = txt(h, svg, 430, 102, '', { class: 'mono' });
          const tB = txt(h, svg, 430, 132, '', { class: 'mono' });
          const tS = txt(h, svg, 430, 172, '', { 'font-size': 15 });
          h.el('circle', { cx: 438, cy: 250, r: 7, fill: 'var(--pos)' }, svg);
          txt(h, svg, 452, 255, 'XOR = +1', { 'font-size': 13 });
          h.el('circle', { cx: 438, cy: 276, r: 7, fill: 'var(--neg)' }, svg);
          txt(h, svg, 452, 281, 'XOR = −1', { 'font-size': 13 });

          /**
           * Finds the offset and orientation that minimise the errors for a line direction theta.
           */
          function bestLine(theta) {
            const nrm = [Math.cos(theta), Math.sin(theta)];
            const v = X.map((p) => nrm[0] * p.x + nrm[1] * p.y);
            const sorted = v.slice().sort((a, b) => a - b);
            const cand = [sorted[0] - 0.3];
            for (let i = 0; i < 3; i++) if (sorted[i + 1] - sorted[i] > 1e-9) cand.push((sorted[i] + sorted[i + 1]) / 2);
            cand.push(sorted[3] + 0.3);
            let best = null;
            cand.forEach((t) => [1, -1].forEach((s) => {
              let errs = 0;
              X.forEach((p, i) => { if ((s * (v[i] - t) >= 0 ? 1 : -1) !== p.c) errs++; });
              if (!best || errs < best.errs) best = { errs, w: [s * nrm[0], s * nrm[1]], b: -s * t };
            }));
            return best;
          }

          let auto = true, tested = 0, minSeen = 4, theta = 0, lastGhost = -1000;

          /**
           * Displays the best line for angle theta and updates the statistics.
           */
          function showAngle(th) {
            const best = bestLine(th);
            plot.setBoundary(best.w, best.b, false);
            minSeen = Math.min(minSeen, best.errs);
            const deg = ((th * 180 / Math.PI) % 180 + 180) % 180;
            h.attr(tT, { text: 'Une droite essaie de séparer…' });
            h.attr(tA, { text: `Angle : ${num(h, deg, 0)}°` });
            h.attr(tE, { text: `Erreurs de cette droite : ${best.errs} / 4`, fill: best.errs === 0 ? 'var(--ok)' : 'var(--bad)' });
            h.attr(tB, { text: `Minimum observé : ${minSeen} / 4 · droites : ${tested}` });
            h.attr(tS, { text: 'Jamais 0 erreur !', fill: 'var(--bad)' });
            return best;
          }

          const slider = h.slider({
            label: 'Angle', min: 0, max: 179, step: 1, value: 0, format: (v) => v + '°',
            onInput: (v) => { auto = false; showAngle(v * Math.PI / 180); }
          });
          h.button('Relancer l\'animation', () => { auto = true; }, 'primary');

          h.loop((t) => {
            if (!auto) return;
            theta = ((t / 10000) * Math.PI) % Math.PI;
            const best = showAngle(theta);
            slider.set(Math.round(theta * 180 / Math.PI) % 180);
            if (t - lastGhost > 220) {
              lastGhost = t;
              tested++;
              const seg = plot.segment(best.w, best.b);
              if (seg) {
                h.el('line', { x1: plot.sx(seg[0][0]), y1: plot.sy(seg[0][1]), x2: plot.sx(seg[1][0]), y2: plot.sy(seg[1][1]), stroke: 'var(--muted)', 'stroke-width': 1.5, opacity: 0.35 }, plot.ghosts);
                if (plot.ghosts.childNodes.length > 34) plot.ghosts.firstChild.remove();
              }
            }
          });
        }
      },
      {
        title: 'Plusieurs couches',
        duration: 20000,
        text: `<p>Si une seule droite ne suffit pas, <b>combinons plusieurs neurones</b>. Le XOR s'écrit
$$\\mathrm{XOR}(x_1,x_2)=\\mathrm{ET}\\big(\\mathrm{OU}(x_1,x_2),\\ \\mathrm{NON\\text{-}ET}(x_1,x_2)\\big)$$
et chacune de ces trois portes est un simple perceptron.</p>
<p>Une première couche calcule $h_1$ (OU) et $h_2$ (NON-ET) ; dans ce nouvel espace $(h_1,h_2)$, les classes deviennent <b>séparables</b> par une droite.</p>
<p>C'est l'idée du <b>perceptron multicouche</b> (MLP), objet du chapitre suivant.</p>`,
        note: 'Les couches cachées <b>transforment l\'espace</b> des entrées pour rendre le problème linéairement séparable. Reste à savoir les entraîner : c\'est la rétropropagation (1986).',
        draw(h) {
          const svg = h.svg(720, 420);
          const step = (a) => (a > 0 ? 1 : 0);
          const nodes = {
            x1: { x: 70, y: 130 }, x2: { x: 70, y: 290 },
            h1: { x: 240, y: 90 }, h2: { x: 240, y: 330 },
            o: { x: 390, y: 210 }
          };
          const edgeDefs = [
            ['x1', 'h1', 1], ['x2', 'h1', 1], ['x1', 'h2', -1], ['x2', 'h2', -1], ['h1', 'o', 1], ['h2', 'o', 1]
          ];
          const edgeEl = edgeDefs.map(([a, b, wv]) => {
            const A = nodes[a], B = nodes[b];
            const col = wv > 0 ? 'var(--pos)' : 'var(--neg)';
            const l = h.el('line', { x1: A.x + 24, y1: A.y, x2: B.x - 24, y2: B.y, stroke: col, 'stroke-width': 3, 'stroke-linecap': 'round', opacity: 0.8 }, svg);
            const mx = h.lerp(A.x + 24, B.x - 24, 0.35), my = h.lerp(A.y, B.y, 0.35);
            h.el('text', { x: mx, y: my - 7, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: col, 'paint-order': 'stroke', stroke: 'var(--surface)', 'stroke-width': 4, text: num(h, wv, 0) }, svg);
            return l;
          });
          const circ = {};
          const val = {};
          Object.keys(nodes).forEach((k) => {
            const nd = nodes[k];
            circ[k] = h.el('circle', { cx: nd.x, cy: nd.y, r: 24, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            val[k] = h.el('text', { x: nd.x, y: nd.y + 5, 'text-anchor': 'middle', 'font-size': 15, class: 'mono', fill: 'var(--ink)' }, svg);
          });
          txt(h, svg, 70 - 0, 100, 'x₁', { 'text-anchor': 'middle', 'font-size': 15 });
          txt(h, svg, 70, 330, 'x₂', { 'text-anchor': 'middle', 'font-size': 15 });
          txt(h, svg, 240, 46, 'h₁ = OU', { 'text-anchor': 'middle', 'font-size': 14 });
          txt(h, svg, 240, 118 + 30, 'b = −0,5', { 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--muted)' });
          txt(h, svg, 240, 292, 'h₂ = NON-ET', { 'text-anchor': 'middle', 'font-size': 14 });
          txt(h, svg, 240, 378, 'b = +1,5', { 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--muted)' });
          txt(h, svg, 390, 170, 'ET', { 'text-anchor': 'middle', 'font-size': 14 });
          txt(h, svg, 390, 262, 'b = −1,5', { 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--muted)' });
          txt(h, svg, 390, 292, 'sortie = XOR', { 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)' });

          const hx = 480, hy = 70, hs = 210;
          const hp = makePlot(h, svg, hx, hy, hs, -0.4, 1.4);
          hp.setBoundary([1, 1], -1.5, false);
          txt(h, svg, hx + hs / 2, hy - 20, 'Espace caché (h₁, h₂)', { 'text-anchor': 'middle', 'font-size': 14 });
          txt(h, svg, hx + hs / 2, hy - 4, 'les classes sont séparables', { 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--ok)' });
          const combos = [[0, 0], [0, 1], [1, 0], [1, 1]];
          const hpts = combos.map(([a, b]) => {
            const o1 = step(a + b - 0.5), o2 = step(-a - b + 1.5), yy = step(o1 + o2 - 1.5);
            return { x: o1, y: o2, c: yy ? 1 : -1 };
          });
          hpts.forEach((p) => h.el('circle', { cx: hp.sx(p.x), cy: hp.sy(p.y), r: 8, fill: colorOf(p.c), stroke: 'var(--surface)', 'stroke-width': 1.5 }, hp.layer));
          const sel = h.el('circle', { r: 14, fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3 }, hp.layer);
          const cap = txt(h, svg, 360, 408, '', { 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--muted)' });
          txt(h, svg, hx + hs / 2, hy + hs + 24, 'h₁ →', { 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' });
          let gen = 0, auto = true, idx = 0;

          /**
           * Animates the signal for the input combination k and updates all node values.
           */
          async function go(k) {
            const my = ++gen;
            idx = k;
            const [a, b] = combos[k];
            const o1 = step(a + b - 0.5), o2 = step(-a - b + 1.5), o = step(o1 + o2 - 1.5);
            const set = (key, v) => {
              h.attr(val[key], { text: v === null ? '' : String(v) });
              h.attr(circ[key], { fill: v === 1 ? 'var(--accent)' : 'var(--surface)', 'fill-opacity': v === 1 ? 0.45 : 1 });
            };
            set('x1', a); set('x2', b); set('h1', null); set('h2', null); set('o', null);
            h.attr(sel, { cx: hp.sx(hpts[k].x), cy: hp.sy(hpts[k].y), opacity: 0 });
            h.attr(cap, { text: `Entrée (${a}, ${b})` });
            const fly = async (pairs) => {
              const toks = pairs.map(([u, v]) => h.el('circle', { cx: nodes[u].x, cy: nodes[u].y, r: 6, fill: 'var(--accent)', stroke: 'var(--surface)', 'stroke-width': 1.5 }, svg));
              await h.tween(550, (q) => {
                pairs.forEach(([u, v], i) => h.attr(toks[i], { cx: h.lerp(nodes[u].x, nodes[v].x, q), cy: h.lerp(nodes[u].y, nodes[v].y, q) }));
              });
              toks.forEach((t) => t.remove());
            };
            await h.sleep(250);
            if (my !== gen) return;
            await fly([['x1', 'h1'], ['x2', 'h1'], ['x1', 'h2'], ['x2', 'h2']]);
            if (my !== gen) return;
            set('h1', o1); set('h2', o2);
            h.attr(cap, { text: `Couche cachée : h₁ = ${o1}, h₂ = ${o2}` });
            h.attr(sel, { opacity: 1 });
            await h.sleep(350);
            if (my !== gen) return;
            await fly([['h1', 'o'], ['h2', 'o']]);
            if (my !== gen) return;
            set('o', o);
            h.attr(cap, { text: `Sortie : XOR(${a}, ${b}) = ${o}` });
          }

          const tg = h.toggle(combos.map(([a, b], k) => [k, `(${a}, ${b})`]), 0, (k) => { auto = false; go(k); });
          tg.addEventListener('pointerdown', () => { auto = false; });
          h.every(3600, () => {
            if (!auto) return;
            const k = (idx + 1) % 4;
            Array.from(tg.children).forEach((bt, i) => bt.setAttribute('aria-pressed', String(i === k)));
            go(k);
          });
          go(0);
        }
      }
    ]
  });
})();
