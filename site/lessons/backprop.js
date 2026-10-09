/**
 * Lesson: back-propagation worked numerically on a 2-2-1 sigmoid network (forward, deltas, gradients, update, training loop, vanishing gradient).
 */
(function () {
  const X = [0.05, 0.10];
  const Y = 0.01;
  const ETA = 0.5;
  const R = 22;
  const IN = [{ x: 70, y: 130 }, { x: 70, y: 290 }];
  const HID = [{ x: 330, y: 130 }, { x: 330, y: 290 }];
  const OUT = { x: 580, y: 210 };
  const HALO = 'paint-order:stroke;stroke:var(--surface);stroke-width:3.5px;stroke-linejoin:round;transition:opacity .3s;';
  const SUB = ['\u2081', '\u2082'];

  /**
   * Logistic sigmoid.
   */
  function sig(a) { return 1 / (1 + Math.exp(-a)); }

  /**
   * Fixed-decimals number with a French decimal comma (module level, used in static texts).
   */
  function n(x, d) { return Number(x).toFixed(d).replace('.', ','); }

  /**
   * Same as n() but with a TeX-safe decimal comma.
   */
  function tn(x, d) { return n(x, d).replace(',', '{,}'); }

  /**
   * Initial parameters of the example network (classic textbook values).
   */
  function makeNet() {
    return { W1: [[0.15, 0.20], [0.25, 0.30]], b1: [0.35, 0.35], W2: [0.40, 0.45], b2: 0.60 };
  }

  /**
   * Deep copy of a network.
   */
  function clone(net) {
    return { W1: net.W1.map((r) => r.slice()), b1: net.b1.slice(), W2: net.W2.slice(), b2: net.b2 };
  }

  /**
   * Forward pass: pre-activations, activations, output and MSE error.
   */
  function forward(net) {
    const a1 = [0, 1].map((j) => net.W1[j][0] * X[0] + net.W1[j][1] * X[1] + net.b1[j]);
    const hh = a1.map(sig);
    const a2 = net.W2[0] * hh[0] + net.W2[1] * hh[1] + net.b2;
    const o = sig(a2);
    return { a1, h: hh, a2, o, E: 0.5 * (Y - o) * (Y - o) };
  }

  /**
   * Backward pass: deltas of the output and hidden cells and all parameter gradients.
   */
  function backward(net, f) {
    const dEdo = -(Y - f.o);
    const gp = f.o * (1 - f.o);
    const dO = dEdo * gp;
    const W2 = f.h.map((hj) => dO * hj);
    const back = net.W2.map((w) => w * dO);
    const gph = f.h.map((hj) => hj * (1 - hj));
    const dH = back.map((s, j) => gph[j] * s);
    const W1 = dH.map((d) => X.map((x) => d * x));
    return { dEdo, gp, dO, back, gph, dH, W2, b2: dO, W1, b1: dH.slice() };
  }

  /**
   * One gradient descent update w <- w - eta * dE/dw.
   */
  function applyStep(net, g, eta) {
    return {
      W1: net.W1.map((r, j) => r.map((w, k) => w - eta * g.W1[j][k])),
      b1: net.b1.map((b, j) => b - eta * g.b1[j]),
      W2: net.W2.map((w, j) => w - eta * g.W2[j]),
      b2: net.b2 - eta * g.b2
    };
  }

  /**
   * Runs the whole worked example once.
   */
  function compute() {
    const net0 = makeNet();
    const f0 = forward(net0);
    const g0 = backward(net0, f0);
    const net1 = applyStep(net0, g0, ETA);
    const f1 = forward(net1);
    return { net0, f0, g0, net1, f1 };
  }

  /**
   * Compares every analytic gradient with a centred finite difference.
   */
  function selfTest(net, g) {
    const eps = 1e-6;
    const loss = (m) => forward(m).E;
    const check = (label, get, set, analytic) => {
      const p = clone(net), m = clone(net);
      set(p, get(p) + eps);
      set(m, get(m) - eps);
      const fd = (loss(p) - loss(m)) / (2 * eps);
      console.assert(Math.abs(fd - analytic) < 1e-8, 'backprop gradient mismatch ' + label, fd, analytic);
    };
    for (let j = 0; j < 2; j++) {
      check('W2' + j, (m) => m.W2[j], (m, v) => { m.W2[j] = v; }, g.W2[j]);
      check('b1' + j, (m) => m.b1[j], (m, v) => { m.b1[j] = v; }, g.b1[j]);
      for (let k = 0; k < 2; k++) check('W1' + j + k, (m) => m.W1[j][k], (m, v) => { m.W1[j][k] = v; }, g.W1[j][k]);
    }
    check('b2', (m) => m.b2, (m, v) => { m.b2 = v; }, g.b2);
  }

  const EX = compute();

  /**
   * Edge width proportional to |w|.
   */
  function edgeW(w) { return Math.min(11, 1 + 5 * Math.abs(w)); }

  /**
   * Point at fraction t along the segment p-q.
   */
  function along(p, q, t) { return { x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t }; }

  /**
   * Segment endpoints shortened so that they stop at the node borders.
   */
  function trim(p, q, r) {
    const dx = q.x - p.x, dy = q.y - p.y, d = Math.hypot(dx, dy) || 1;
    return [{ x: p.x + (dx / d) * r, y: p.y + (dy / d) * r }, { x: q.x - (dx / d) * r, y: q.y - (dy / d) * r }];
  }

  /**
   * Adds an arrow marker with a custom colour token to an SVG.
   */
  function addMarker(h, svg, id, token) {
    const m = h.el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, svg);
    h.el('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--' + token + ')' }, m);
  }

  /**
   * Scientific notation with a decimal comma, e.g. 2,5e-1.
   */
  function sci(x) {
    if (x === 0) return '0';
    return x.toExponential(1).replace('e+', 'e').replace('.', ',');
  }

  /**
   * Superscript rendering of an integer exponent.
   */
  function sup(e) {
    const map = { '-': '\u207B', 0: '\u2070', 1: '\u00B9', 2: '\u00B2', 3: '\u00B3', 4: '\u2074', 5: '\u2075', 6: '\u2076', 7: '\u2077', 8: '\u2078', 9: '\u2079' };
    return String(e).split('').map((c) => map[c]).join('');
  }

  /**
   * Builds the 2-2-1 network drawing (edges, nodes, empty value slots) and returns its handles.
   * opts.parent: group to draw into, opts.bare: no texts, no legend, no strip.
   */
  function buildNet(h, svg, opts) {
    opts = opts || {};
    const root = opts.parent || svg;
    const bare = !!opts.bare;
    const v = { e1: [[null, null], [null, null]], e2: [null, null], t: {}, strip: null };
    const eg = h.el('g', {}, root);
    const ng = h.el('g', {}, root);
    const tg = h.el('g', {}, root);

    [0, 1].forEach((j) => {
      [0, 1].forEach((k) => {
        v.e1[j][k] = h.el('line', { x1: IN[k].x, y1: IN[k].y, x2: HID[j].x, y2: HID[j].y, 'stroke-linecap': 'round' }, eg);
      });
      v.e2[j] = h.el('line', { x1: HID[j].x, y1: HID[j].y, x2: OUT.x, y2: OUT.y, 'stroke-linecap': 'round' }, eg);
    });

    v.nodes = { inp: [], hid: [], out: null };
    const node = (p, sym) => {
      const c = h.el('circle', { cx: p.x, cy: p.y, r: R, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, ng);
      if (!bare) h.el('text', { x: p.x, y: p.y, 'text-anchor': 'middle', style: 'dominant-baseline:central;font-size:15px;font-style:italic', text: sym }, ng);
      return c;
    };
    [0, 1].forEach((k) => { v.nodes.inp[k] = node(IN[k], 'x' + SUB[k]); });
    [0, 1].forEach((j) => { v.nodes.hid[j] = node(HID[j], 'h' + SUB[j]); });
    v.nodes.out = node(OUT, 'o');

    if (!bare) {
      h.el('rect', { x: 14, y: 410, width: 692, height: 54, rx: 10, fill: 'var(--surface-2)' }, root);
      [[IN[0].x, 'Entrée'], [HID[0].x, 'Couche cachée L\u22121'], [OUT.x, 'Sortie L']].forEach(([x, s]) => {
        h.el('text', { x, y: 396, 'text-anchor': 'middle', class: 'muted', style: 'font-size:12px', text: s }, root);
      });
    }

    const mk = (key, x, y, anchor, size) => {
      v.t[key] = h.el('text', { x, y, 'text-anchor': anchor, class: 'mono', style: HALO + 'font-size:' + (size || 12) + 'px;opacity:0' }, tg);
    };
    if (!bare) {
      mk('x0', IN[0].x, IN[0].y - 32, 'middle');
      mk('x1', IN[1].x, IN[1].y + 44, 'middle');
      const ys = [[68, 82, 96], [328, 342, 356]];
      [0, 1].forEach((j) => {
        mk('hb' + j, HID[j].x, ys[j][0], 'middle');
        mk('ha' + j, HID[j].x, ys[j][1], 'middle');
        mk('hh' + j, HID[j].x, ys[j][2], 'middle');
        mk('hd' + j, HID[j].x, j === 0 ? 174 : 256, 'middle');
        [0, 1].forEach((k) => {
          const pa = along(IN[k], HID[j], 0.27), pb = along(IN[k], HID[j], 0.73);
          mk('wa' + j + k, pa.x, pa.y + 4, 'middle');
          mk('wb' + j + k, pb.x, pb.y + 4, 'middle');
        });
        const pa = along(HID[j], OUT, 0.3), pb = along(HID[j], OUT, 0.7);
        mk('va' + j, pa.x, pa.y + 4, 'middle');
        mk('vb' + j, pb.x, pb.y + 4, 'middle');
      });
      mk('ob', 608, 184, 'start');
      mk('oa', 608, 200, 'start');
      mk('oo', 608, 216, 'start');
      mk('od', 608, 232, 'start');
      mk('py', 712, 24, 'end', 13);
      mk('pE', 712, 44, 'end', 13);
      mk('pE2', 712, 64, 'end', 13);
    }

    /**
     * Writes a value into a slot and makes it visible.
     */
    v.set = (key, str, color, size) => {
      const e = v.t[key];
      if (!e) return;
      e.textContent = str;
      e.style.opacity = 1;
      e.style.fill = color || 'var(--ink)';
      if (size) e.style.fontSize = size + 'px';
    };
    v.hide = (key) => { if (v.t[key]) v.t[key].style.opacity = 0; };
    v.hideMatch = (prefix) => { Object.keys(v.t).forEach((k) => { if (k.indexOf(prefix) === 0) v.hide(k); }); };

    /**
     * Colours and sizes every edge from the weights of net.
     */
    v.paint = (net) => {
      [0, 1].forEach((j) => {
        [0, 1].forEach((k) => {
          const w = net.W1[j][k];
          h.attr(v.e1[j][k], { stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': edgeW(w), opacity: 0.85 });
        });
        const w = net.W2[j];
        h.attr(v.e2[j], { stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': edgeW(w), opacity: 0.85 });
      });
    };

    /**
     * Temporarily recolours an edge (used for signals flowing along it).
     */
    v.glow = (edge, token) => { h.attr(edge, { stroke: 'var(--' + token + ')', opacity: 1 }); };

    /**
     * Highlights (or resets) a node outline.
     */
    v.node = (el, token) => {
      h.attr(el, { stroke: token ? 'var(--' + token + ')' : 'var(--ink)', 'stroke-width': token ? 3.5 : 1.5 });
    };

    /**
     * Shows the weights in the first slot of each edge.
     */
    v.weights = (net, d) => {
      const dd = d === undefined ? 3 : d;
      [0, 1].forEach((j) => {
        [0, 1].forEach((k) => v.set('wa' + j + k, h.fmt(net.W1[j][k], dd)));
        v.set('va' + j, h.fmt(net.W2[j], dd));
      });
    };

    v.showIn = () => { [0, 1].forEach((k) => v.set('x' + k, 'x' + SUB[k] + '=' + h.fmt(X[k], 2))); };
    v.showHid = (net, f, j, which) => {
      const w = which || 'bah';
      if (w.indexOf('b') >= 0) v.set('hb' + j, 'b=' + h.fmt(net.b1[j], 3), 'var(--muted)');
      if (w.indexOf('a') >= 0) v.set('ha' + j, 'a=' + h.fmt(f.a1[j], 3), 'var(--accent)');
      if (w.indexOf('h') >= 0) v.set('hh' + j, 'h=' + h.fmt(f.h[j], 3), 'var(--accent)');
    };
    v.showOut = (net, f, which) => {
      const w = which || 'bao';
      if (w.indexOf('b') >= 0) v.set('ob', 'b=' + h.fmt(net.b2, 3), 'var(--muted)');
      if (w.indexOf('a') >= 0) v.set('oa', 'a=' + h.fmt(f.a2, 3), 'var(--accent)');
      if (w.indexOf('o') >= 0) v.set('oo', 'o=' + h.fmt(f.o, 3), 'var(--accent)');
    };
    v.showErr = (f) => {
      v.set('py', 'cible y = ' + h.fmt(Y, 2), 'var(--ink)', 13);
      v.set('pE', 'E = ' + h.fmt(f.E, 4), 'var(--ink)', 13);
    };
    v.showForward = (net, f) => {
      v.showIn();
      [0, 1].forEach((j) => v.showHid(net, f, j));
      v.showOut(net, f);
      v.showErr(f);
    };
    v.showDeltaOut = (g) => v.set('od', '\u03B4=' + h.fmt(g.dO, 4), 'var(--accent-2)');
    v.showGradOut = (g) => { [0, 1].forEach((j) => v.set('vb' + j, '\u2202=' + h.fmt(g.W2[j], 4), 'var(--accent-2)')); };
    v.showDeltaHid = (g) => { [0, 1].forEach((j) => v.set('hd' + j, '\u03B4=' + h.fmt(g.dH[j], 4), 'var(--accent-2)')); };
    v.showGradHid = (g) => { [0, 1].forEach((j) => [0, 1].forEach((k) => v.set('wb' + j + k, '\u2202=' + h.fmt(g.W1[j][k], 4), 'var(--accent-2)'))); };

    /**
     * Moves a dot between two node centres (shortened to the node borders).
     */
    v.dot = (p, q, token, ms) => {
      const [a, b] = trim(p, q, R);
      const c = h.el('circle', { r: 5.5, cx: a.x, cy: a.y, fill: 'var(--' + token + ')', stroke: 'var(--surface)', 'stroke-width': 1.5 }, tg);
      return h.tween(ms, (e) => h.attr(c, { cx: h.lerp(a.x, b.x, e), cy: h.lerp(a.y, b.y, e) })).then(() => c.remove());
    };

    /**
     * Writes HTML+TeX into the bottom caption strip.
     */
    v.say = (html) => {
      if (bare) return;
      if (!v.strip) v.strip = h.label(svg, 22, 412, 676, 50, '', 'center');
      v.strip.innerHTML = html;
      h.typeset(v.strip);
    };
    return v;
  }

  Course.register({
    id: 'backprop',
    order: 6,
    title: 'Rétropropagation (backprop)',
    summary: 'Le calcul du gradient couche par couche, <strong>entièrement déroulé à la main</strong> sur un petit réseau 2-2-1 : passe avant, erreurs δ, gradients, mise à jour, boucle d\'entraînement, et le problème du gradient qui s\'évanouit.',
    pdfPages: '22-25',

    init(shared) {
      const ex = compute();
      Object.assign(shared, ex);
      selfTest(ex.net0, ex.g0);
      console.assert(ex.f1.E < ex.f0.E, 'one gradient step should reduce the error');
    },

    steps: [
      {
        title: 'Le but : le gradient',
        duration: 12000,
        text: String.raw`<p>Entraîner un réseau, c'est faire descendre l'erreur $E$ en modifiant <em>tous</em> les paramètres. Pour cela il faut le gradient $\partial E/\partial w$ de <strong>chaque</strong> poids.</p>
<p>Pour un poids de la dernière couche, c'est direct. Pour une couche cachée, le poids est « loin » de l'erreur : il agit sur $a$, qui agit sur $o$, qui agit sur $E$. La <strong>règle de dérivation en chaîne</strong> permet de s'en sortir : on <em>multiplie les dérivées locales</em> le long du chemin.</p>
$$\frac{\partial E}{\partial w}=\frac{\partial E}{\partial o}\cdot\frac{\partial o}{\partial a}\cdot\frac{\partial a}{\partial w}$$
<p>La rétropropagation calcule ces produits <em>de droite à gauche</em> en réutilisant les facteurs déjà obtenus. Regardez la chaîne se remplir avec les nombres de notre exemple.</p>`,
        note: String.raw`Notation du cours : $w_{ij}$ relie la $j$-ème cellule de la couche cachée $L-1$ à la $i$-ème cellule de sortie $L$ ; $a_i=\sum_j w_{ij}h_j+b_i$, $o_i=g(a_i)$ avec $g$ la sigmoïde et $g'(a)=g(a)(1-g(a))$.`,
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 420);
          addMarker(h, svg, 'arrow-a2', 'accent-2');
          const bx = [30, 215, 400, 585], bw = 105, by = 105, bh = 60;
          const caps = ['poids', 'pré-activation', 'sortie', 'erreur'];
          const syms = ['$w_{11}$', '$a$', '$o$', '$E$'];
          const boxes = bx.map((x, i) => {
            const g = h.el('g', { style: 'opacity:0;transition:opacity .5s' }, svg);
            h.el('rect', { x, y: by, width: bw, height: bh, rx: 12, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
            h.label(g, x, by, bw, bh, '<span style="font-size:20px">' + syms[i] + '</span>', 'center');
            h.el('text', { x: x + bw / 2, y: by + bh + 20, 'text-anchor': 'middle', class: 'muted', style: 'font-size:12px', text: caps[i] }, g);
            return g;
          });
          const syms2 = [
            String.raw`$\partial a/\partial w$<br>$=h_1$`,
            String.raw`$\partial o/\partial a$<br>$=g'(a)$`,
            String.raw`$\partial E/\partial o$<br>$=-(y-o)$`
          ];
          const vals = [S.f0.h[0], S.g0.gp, S.g0.dEdo];
          const fwd = [], bwd = [], nums = [], names = [];
          for (let i = 0; i < 3; i++) {
            const x0 = bx[i] + bw + 6, x1 = bx[i + 1] - 6, cx = (x0 + x1) / 2;
            fwd[i] = h.el('line', { x1: x0, y1: 135, x2: x1, y2: 135, stroke: 'var(--muted)', 'stroke-width': 2, 'marker-end': 'url(#arrow-muted)', style: 'opacity:0;transition:opacity .5s' }, svg);
            names[i] = h.label(svg, cx - 44, 62, 88, 52, syms2[i], 'center');
            bwd[i] = h.el('line', { x1: x1, y1: 200, x2: x0, y2: 200, stroke: 'var(--accent-2)', 'stroke-width': 3, 'marker-end': 'url(#arrow-a2)', style: 'opacity:0;transition:opacity .4s' }, svg);
            nums[i] = h.el('text', { x: cx, y: 224, 'text-anchor': 'middle', class: 'mono', style: 'font-size:14px;fill:var(--accent-2);opacity:0;transition:opacity .4s', text: h.fmt(vals[i], 3) }, svg);
          }
          h.el('text', { x: 360, y: 252, 'text-anchor': 'middle', class: 'muted', style: 'font-size:12px', text: '\u2190 le signal d\'erreur remonte, on multiplie au fur et à mesure' }, svg);
          h.el('rect', { x: 20, y: 290, width: 680, height: 90, rx: 12, fill: 'var(--surface-2)' }, svg);
          const strip = h.label(svg, 30, 296, 660, 78, '', 'center');
          const say = (s) => { strip.innerHTML = s; h.typeset(strip); };
          (async () => {
            say(String.raw`<span class="muted">$w_{11}$ (poids $h_1\to o$) influence $a$, puis $o$, puis $E$</span>`);
            for (let i = 0; i < 4; i++) {
              boxes[i].style.opacity = 1;
              if (i < 3) fwd[i].style.opacity = 1;
              await h.sleep(450);
            }
            await h.sleep(900);
            const d3 = S.g0.dEdo, d2 = S.g0.dO, d1 = S.g0.W2[0];
            const steps = [
              [2, String.raw`$\dfrac{\partial E}{\partial o}=-(y-o)=${tn(d3, 3)}$`],
              [1, String.raw`$\dfrac{\partial E}{\partial a}=\dfrac{\partial E}{\partial o}\cdot g'(a)=${tn(d3, 3)}\times${tn(S.g0.gp, 3)}=${tn(d2, 4)}$`],
              [0, String.raw`$\dfrac{\partial E}{\partial w_{11}}=${tn(d2, 4)}\times h_1=${tn(d2, 4)}\times${tn(S.f0.h[0], 3)}=${tn(d1, 4)}$`]
            ];
            for (const [i, s] of steps) {
              bwd[i].style.opacity = 1;
              nums[i].style.opacity = 1;
              say(s);
              await h.sleep(2300);
            }
          })();
        }
      },

      {
        title: 'Passe avant',
        duration: 12000,
        text: String.raw`<p>Notre mini-réseau : <strong>2 entrées → 2 cellules cachées sigmoïdes → 1 sortie sigmoïde</strong>, avec un biais par cellule. Un seul exemple : $x=(${tn(X[0], 2)},\,${tn(X[1], 2)})$ et cible $y=${tn(Y, 2)}$. Les poids de départ sont fixés (l'épaisseur du trait ∝ $|w|$, bleu = positif, orange = négatif).</p>
<p>D'abord la <strong>passe avant</strong> : on propage les valeurs de gauche à droite, couche par couche.</p>
$$a_j=\sum_k w_{jk}x_k+b_j,\quad h_j=g(a_j)\qquad a=\sum_j w_{j}h_j+b,\quad o=g(a)$$
<p>Résultat : $o=${tn(EX.f0.o, 3)}$, très loin de la cible. L'erreur quadratique vaut $E=\tfrac12(y-o)^2=${tn(EX.f0.E, 4)}$.</p>`,
        note: 'Il faut <strong>garder en mémoire</strong> les $a$ et les $h$ de la passe avant : la passe arrière en a besoin.',
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          v.paint(S.net0);
          v.weights(S.net0);
          v.say(String.raw`<span>Poids initiaux affichés sur les arêtes, biais $b$ près des cellules. Lancement de la passe avant…</span>`);
          (async () => {
            await h.sleep(500);
            v.showIn();
            v.set('py', 'cible y = ' + h.fmt(Y, 2), 'var(--ink)', 13);
            await h.sleep(800);
            v.say(String.raw`Couche cachée : $a_j=\sum_k w_{jk}x_k+b_j$, puis $h_j=g(a_j)=\dfrac{1}{1+e^{-a_j}}$`);
            const jobs = [];
            [0, 1].forEach((j) => [0, 1].forEach((k) => { v.glow(v.e1[j][k], 'accent'); jobs.push(v.dot(IN[k], HID[j], 'accent', 1100)); }));
            await Promise.all(jobs);
            v.paint(S.net0);
            for (let j = 0; j < 2; j++) {
              v.showHid(S.net0, S.f0, j, 'b');
              await h.sleep(250);
              v.showHid(S.net0, S.f0, j, 'a');
              await h.sleep(350);
              v.showHid(S.net0, S.f0, j, 'h');
              v.node(v.nodes.hid[j], 'accent');
              await h.sleep(500);
            }
            v.say(String.raw`Sortie : $a=\sum_j w_{j}h_j+b=${tn(S.f0.a2, 3)}$, puis $o=g(a)=${tn(S.f0.o, 3)}$`);
            await h.sleep(700);
            const jobs2 = [];
            [0, 1].forEach((j) => { v.glow(v.e2[j], 'accent'); jobs2.push(v.dot(HID[j], OUT, 'accent', 1100)); });
            await Promise.all(jobs2);
            v.paint(S.net0);
            v.showOut(S.net0, S.f0, 'b');
            await h.sleep(250);
            v.showOut(S.net0, S.f0, 'a');
            await h.sleep(350);
            v.showOut(S.net0, S.f0, 'o');
            v.node(v.nodes.out, 'accent');
            await h.sleep(600);
            v.showErr(S.f0);
            v.say(String.raw`Erreur : $E=\tfrac12(y-o)^2=\tfrac12(${tn(Y, 2)}-${tn(S.f0.o, 3)})^2=${tn(S.f0.E, 4)}$ \u2014 il faut corriger les poids.`.replace('\\u2014', '\u2014'));
          })();
        }
      },

      {
        title: 'Erreur de sortie',
        duration: 13000,
        text: String.raw`<p>On commence par la dernière couche, la plus simple. Avec l'erreur quadratique, la dérivée par rapport à la sortie est
$\dfrac{\partial E}{\partial o}=-(y-o)$ : si $o>y$ elle est positive (la sortie est trop grande).</p>
<p>On la multiplie par la pente de la sigmoïde pour obtenir le <strong>delta de sortie</strong> :</p>
$$\delta=\frac{\partial E}{\partial a}=-(y-o)\,g'(a),\qquad g'(a)=o(1-o)$$
<p>Un delta donne ensuite directement le gradient de chaque poids entrant : le poids $w_{ij}$ relie $h_j$ à $a_i$, donc</p>
$$\frac{\partial E}{\partial w_{ij}}=\delta_i\,h_j\qquad\frac{\partial E}{\partial b_i}=\delta_i$$`,
        note: String.raw`Ici $\delta=${tn(EX.g0.dO, 4)}>0$ : augmenter $a$ augmenterait l'erreur, donc la descente de gradient va <em>diminuer</em> les poids vers la sortie.`,
        check: {
          q: String.raw`Avec $o=${tn(EX.f0.o, 3)}$ et $y=${tn(Y, 2)}$, quelle est la valeur du delta de sortie $\delta=-(y-o)\,o(1-o)$ ?`,
          choices: ['$' + tn(EX.g0.dEdo, 3) + '$ (la dérivée $\\partial E/\\partial o$ seule)', '$' + tn(EX.g0.dO, 4) + '$', '$' + tn(EX.g0.gp, 3) + '$ (la pente $g\'(a)$ seule)'],
          answer: 1,
          explain: 'Il faut multiplier les deux facteurs : ' + n(EX.g0.dEdo, 3) + ' × ' + n(EX.g0.gp, 3) + ' = ' + n(EX.g0.dO, 4) + '.'
        },
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          v.paint(S.net0);
          v.weights(S.net0);
          v.showForward(S.net0, S.f0);
          (async () => {
            v.node(v.nodes.out, 'accent-2');
            v.say(String.raw`Dérivée de l'erreur : $\dfrac{\partial E}{\partial o}=-(y-o)=-(${tn(Y, 2)}-${tn(S.f0.o, 3)})=${tn(S.g0.dEdo, 3)}$`);
            await h.sleep(2600);
            v.say(String.raw`Pente de la sigmoïde : $g'(a)=o(1-o)=${tn(S.f0.o, 3)}\times${tn(1 - S.f0.o, 3)}=${tn(S.g0.gp, 3)}$`);
            await h.sleep(2600);
            v.say(String.raw`$\delta=\dfrac{\partial E}{\partial a}=${tn(S.g0.dEdo, 3)}\times${tn(S.g0.gp, 3)}=${tn(S.g0.dO, 4)}$`);
            v.showDeltaOut(S.g0);
            await h.sleep(2600);
            for (let j = 0; j < 2; j++) {
              v.glow(v.e2[j], 'accent-2');
              v.set('vb' + j, '\u2202=' + h.fmt(S.g0.W2[j], 4), 'var(--accent-2)');
              v.say(String.raw`$\dfrac{\partial E}{\partial w_{1${j + 1}}}=\delta\cdot h_${j + 1}=${tn(S.g0.dO, 4)}\times${tn(S.f0.h[j], 3)}=${tn(S.g0.W2[j], 4)}$`);
              await h.sleep(2200);
              v.paint(S.net0);
            }
            v.say(String.raw`Et pour le biais : $\dfrac{\partial E}{\partial b}=\delta=${tn(S.g0.dO, 4)}$`);
          })();
        }
      },

      {
        title: 'Erreur cachée',
        duration: 13000,
        text: String.raw`<p>Les cellules cachées n'ont pas de « cible », mais elles ont une part de responsabilité dans l'erreur de sortie. On <strong>renvoie l'erreur en arrière</strong> le long des arêtes, pondérée par les poids :</p>
$$\delta_j=\frac{\partial E}{\partial a_j}=g'(a_j)\sum_i w_{ij}\,\delta_i,\qquad g'(a_j)=h_j(1-h_j)$$
<p>Pourquoi cette somme ? Un changement de $h_j$ modifie <em>chaque</em> $a_i$ de la couche suivante (de $w_{ij}$ fois), donc l'effet total sur $E$ est la somme $\sum_i \delta_i w_{ij}$ (chain rule). Ici, il n'y a qu'une seule sortie : la somme n'a qu'un terme.</p>
<p>Observez les points violets remonter de la sortie vers chaque cellule cachée.</p>`,
        note: 'Même schéma que pour la sortie : <strong>delta = pente de l\'activation × erreur reçue</strong>. La différence est que l\'erreur reçue vient maintenant de la couche suivante.',
        check: {
          q: String.raw`Dans $\delta_j=g'(a_j)\sum_i w_{ij}\delta_i$, que se passe-t-il si le poids $w_{ij}=0$ ?`,
          choices: ['La cellule $j$ ne reçoit rien de la cellule $i$ : elle n\'influence pas $a_i$, donc pas cette part de l\'erreur', 'L\'erreur de la cellule $j$ devient infinie', 'Le delta $\\delta_i$ est automatiquement nul'],
          answer: 0,
          explain: 'Un poids nul coupe l\'arête : $h_j$ n\'a aucun effet sur $a_i$, donc le terme $w_{ij}\\delta_i$ disparaît de la somme.'
        },
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          v.paint(S.net0);
          v.weights(S.net0);
          v.showForward(S.net0, S.f0);
          v.showDeltaOut(S.g0);
          v.say(String.raw`L'erreur de sortie $\delta=${tn(S.g0.dO, 4)}$ va être renvoyée vers les cellules cachées.`);
          (async () => {
            await h.sleep(1500);
            for (let j = 0; j < 2; j++) {
              v.glow(v.e2[j], 'accent-2');
              v.say(String.raw`Retour vers $h_${j + 1}$ : $w_{1${j + 1}}\,\delta=${tn(S.net0.W2[j], 3)}\times${tn(S.g0.dO, 4)}=${tn(S.g0.back[j], 4)}$`);
              await v.dot(OUT, HID[j], 'accent-2', 1400);
              v.set('vb' + j, 'w\u00B7\u03B4=' + h.fmt(S.g0.back[j], 4), 'var(--accent-2)');
              await h.sleep(500);
              v.say(String.raw`Puis on multiplie par la pente : $\delta_${j + 1}=h_${j + 1}(1-h_${j + 1})\cdot w_{1${j + 1}}\delta=${tn(S.g0.gph[j], 3)}\times${tn(S.g0.back[j], 4)}=${tn(S.g0.dH[j], 4)}$`);
              v.set('hd' + j, 'δ=' + h.fmt(S.g0.dH[j], 4), 'var(--accent-2)');
              v.node(v.nodes.hid[j], 'accent-2');
              await h.sleep(2800);
            }
            v.paint(S.net0);
            v.say(String.raw`Les deltas cachés sont plus petits que le delta de sortie : l'erreur est <em>diluée</em> à chaque couche remontée.`);
          })();
        }
      },

      {
        title: 'Gradients couche 1',
        duration: 12000,
        text: String.raw`<p>Maintenant que chaque cellule cachée connaît son delta $\delta_j$, le gradient d'un poids de la première couche a exactement la même forme que pour la sortie : <em>delta de la cellule d'arrivée × valeur qui entre par l'arête</em>.</p>
$$\frac{\partial E}{\partial w_{jk}}=\delta_j\,x_k\qquad\frac{\partial E}{\partial b_j}=\delta_j$$
<p>On a maintenant le gradient de <strong>tous</strong> les paramètres, obtenu avec un seul aller-retour dans le réseau.</p>
<p>Ces gradients sont petits (≈ $10^{-3}$) : les entrées $x_k$ sont petites et la sigmoïde est peu pentue. On les affiche avec 4 décimales.</p>`,
        note: 'Les gradients ne sont pas égaux au delta : ils sont multipliés par la valeur d\'<em>entrée</em> de l\'arête ($x_k$ ou $h_j$). Une entrée nulle ⇒ gradient nul.',
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          v.paint(S.net0);
          v.weights(S.net0);
          v.showForward(S.net0, S.f0);
          v.showDeltaOut(S.g0);
          v.showGradOut(S.g0);
          v.showDeltaHid(S.g0);
          v.say(String.raw`Deltas disponibles : $\delta_1=${tn(S.g0.dH[0], 4)}$, $\delta_2=${tn(S.g0.dH[1], 4)}$. Calculons $\partial E/\partial w_{jk}=\delta_j x_k$.`);
          (async () => {
            await h.sleep(1600);
            for (let j = 0; j < 2; j++) {
              const jobs = [];
              [0, 1].forEach((k) => { v.glow(v.e1[j][k], 'accent-2'); jobs.push(v.dot(HID[j], IN[k], 'accent-2', 1000)); });
              v.node(v.nodes.hid[j], 'accent-2');
              await Promise.all(jobs);
              for (let k = 0; k < 2; k++) {
                v.set('wb' + j + k, '\u2202=' + h.fmt(S.g0.W1[j][k], 4), 'var(--accent-2)');
                v.say(String.raw`$\dfrac{\partial E}{\partial w_{${j + 1}${k + 1}}}=\delta_${j + 1}\,x_${k + 1}=${tn(S.g0.dH[j], 4)}\times${tn(X[k], 2)}=${tn(S.g0.W1[j][k], 4)}$`);
                await h.sleep(1500);
              }
              v.paint(S.net0);
              v.node(v.nodes.hid[j], null);
            }
            v.say(String.raw`Tous les gradients sont connus : $\partial E/\partial b_j=\delta_j$ pour les biais cachés, $\partial E/\partial b=${tn(S.g0.b2, 4)}$ pour la sortie.`);
          })();
        }
      },

      {
        title: 'Mise à jour des poids',
        duration: 12000,
        text: String.raw`<p>Il ne reste plus qu'à faire un pas de <strong>descente de gradient</strong>, dans la direction opposée au gradient, avec un pas $\eta$ (<em>learning rate</em>) :</p>
$$w\leftarrow w-\eta\,\frac{\partial E}{\partial w}\qquad b\leftarrow b-\eta\,\frac{\partial E}{\partial b}$$
<p>Avec $\eta=${tn(ETA, 1)}$, par exemple $w_{11}$ de la sortie passe de ${tn(EX.net0.W2[0], 4)}$ à ${tn(EX.net1.W2[0], 4)}$. Les poids reliés aux gros gradients bougent le plus : l'arête de sortie s'amincit visiblement.</p>
<p>Essayez de changer $\eta$ avec le curseur : trop petit, on avance à peine ; trop grand, on risque de dépasser le minimum.</p>`,
        note: String.raw`Le signe compte : $\partial E/\partial w>0$ ⇒ $E$ augmente quand $w$ augmente ⇒ on <em>diminue</em> $w$.`,
        check: {
          q: String.raw`Si $\partial E/\partial w=+0{,}08$ et $\eta=0{,}5$, comment évolue $w$ ?`,
          choices: ['$w$ augmente de $0{,}04$', '$w$ diminue de $0{,}04$', '$w$ ne change pas'],
          answer: 1,
          explain: 'On soustrait $\\eta\\,\\partial E/\\partial w = 0{,}04$ : le gradient positif indique que $E$ monte avec $w$, on recule donc.'
        },
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          let eta = ETA;
          v.paint(S.net0);
          v.weights(S.net0);
          v.showForward(S.net0, S.f0);
          v.showDeltaOut(S.g0);
          v.showDeltaHid(S.g0);
          v.showGradOut(S.g0);
          v.showGradHid(S.g0);
          const mix = (a, b, p) => ({
            W1: a.W1.map((r, j) => r.map((w, k) => h.lerp(w, b.W1[j][k], p))),
            b1: a.b1.map((x, j) => h.lerp(x, b.b1[j], p)),
            W2: a.W2.map((w, j) => h.lerp(w, b.W2[j], p)),
            b2: h.lerp(a.b2, b.b2, p)
          });
          /**
           * Draws the interpolated network between the old and the updated parameters.
           */
          const show = (target, p) => {
            const cur = mix(S.net0, target, p);
            v.paint(cur);
            [0, 1].forEach((j) => {
              [0, 1].forEach((k) => v.set('wa' + j + k, h.fmt(S.net0.W1[j][k], 4) + '\u2192' + h.fmt(cur.W1[j][k], 4), 'var(--ink)', 11));
              v.set('va' + j, h.fmt(S.net0.W2[j], 4) + '\u2192' + h.fmt(cur.W2[j], 4), 'var(--ink)', 11);
              v.set('hb' + j, 'b=' + h.fmt(S.net0.b1[j], 3) + '\u2192' + h.fmt(cur.b1[j], 3), 'var(--muted)');
            });
            v.set('ob', 'b=' + h.fmt(S.net0.b2, 3) + '\u2192' + h.fmt(cur.b2, 3), 'var(--muted)');
          };
          const say = () => v.say(String.raw`$w\leftarrow w-\eta\,\dfrac{\partial E}{\partial w}$ avec $\eta=${tn(eta, 2)}$ \u2014 ex. $w_{11}$ : $${tn(S.net0.W2[0], 4)}-${tn(eta, 2)}\times${tn(S.g0.W2[0], 4)}=${tn(S.net0.W2[0] - eta * S.g0.W2[0], 4)}$`.replace('\\u2014', '\u2014'));
          say();
          h.slider({
            label: '$\\eta$', min: 0.05, max: 3, step: 0.05, value: ETA, decimals: 2,
            onInput(val) {
              eta = val;
              show(applyStep(S.net0, S.g0, eta), 1);
              say();
            }
          });
          (async () => {
            await h.sleep(1500);
            await h.tween(2500, (p) => show(applyStep(S.net0, S.g0, eta), p));
          })();
        }
      },

      {
        title: 'Vérification',
        duration: 11000,
        text: String.raw`<p>Refaisons une passe avant avec les poids mis à jour. Si tout est correct, l'erreur doit avoir <strong>diminué</strong> :</p>
$$E:\ ${tn(EX.f0.E, 4)}\ \longrightarrow\ ${tn(EX.f1.E, 4)}$$
<p>La sortie est passée de $${tn(EX.f0.o, 3)}$ à $${tn(EX.f1.o, 3)}$, un peu plus proche de la cible $y=${tn(Y, 2)}$. Un seul pas ne suffit pas : il faut répéter (étape suivante).</p>
<p>On a aussi vérifié le calcul : les gradients de la rétropropagation coïncident avec une <strong>différence finie</strong> $\dfrac{E(w+\varepsilon)-E(w-\varepsilon)}{2\varepsilon}$ (testée au chargement de la leçon, voir la console).</p>
<p>Curseur $\eta$ : essayez de grandes valeurs, le pas peut dépasser le fond de la vallée.</p>`,
        note: 'Test classique de débogage : comparer le gradient de la backprop à une différence finie numérique (<em>gradient checking</em>).',
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 470);
          const v = buildNet(h, svg);
          const out = h.readout('');
          /**
           * Shows the network after an update with the given learning rate.
           */
          const apply = async (eta, animate) => {
            const net1 = applyStep(S.net0, S.g0, eta);
            const f1 = forward(net1);
            v.paint(net1);
            v.weights(net1);
            if (animate) {
              const jobs = [];
              [0, 1].forEach((j) => [0, 1].forEach((k) => jobs.push(v.dot(IN[k], HID[j], 'accent', 900))));
              await Promise.all(jobs);
              const jobs2 = [];
              [0, 1].forEach((j) => jobs2.push(v.dot(HID[j], OUT, 'accent', 900)));
              await Promise.all(jobs2);
            }
            v.showForward(net1, f1);
            v.set('pE', 'E avant = ' + h.fmt(S.f0.E, 4), 'var(--ink)', 13);
            v.set('pE2', 'E après = ' + h.fmt(f1.E, 4), f1.E < S.f0.E ? 'var(--ok)' : 'var(--bad)', 13);
            v.say(String.raw`$E$ : $${tn(S.f0.E, 4)}\ \to\ ${tn(f1.E, 4)}$ \u2014 ${f1.E < S.f0.E ? 'l\'erreur a diminué' : 'l\'erreur a augmenté (pas trop grand)'}`.replace('\\u2014', '\u2014'));
            out('$\\eta=' + tn(eta, 2) + '$ : variation de $E$ = ' + h.fmt(100 * (f1.E - S.f0.E) / S.f0.E, 1) + ' %');
          };
          v.paint(S.net0);
          v.weights(S.net0);
          v.showForward(S.net0, S.f0);
          v.say(String.raw`Avant la mise à jour : $E=${tn(S.f0.E, 4)}$. Nouvelle passe avant avec les poids mis à jour…`);
          h.slider({
            label: '$\\eta$', min: 0.05, max: 8, step: 0.05, value: ETA, decimals: 2,
            onInput(val) { apply(val, false); }
          });
          (async () => {
            await h.sleep(1500);
            await apply(ETA, true);
          })();
        }
      },

      {
        title: 'Boucle d\'entraînement',
        duration: 15000,
        text: String.raw`<p>L'<strong>entraînement</strong> répète ces trois temps : passe avant, passe arrière, mise à jour. À chaque itération l'erreur baisse (courbe de droite) et les poids évoluent (épaisseur et couleur des arêtes).</p>
<p>Avec un seul exemple, c'est une descente de gradient simple ; avec beaucoup d'exemples on moyenne les gradients sur un <em>minibatch</em> (SGD).</p>
<p>Appuyez sur « Itérer ». Réglez $\eta$ : trop faible ⇒ très lent ; plus grand ⇒ plus rapide, mais attention à l'instabilité. Passez en échelle log pour voir la décroissance à long terme.</p>`,
        note: 'La courbe d\'erreur est la visualisation de référence : elle doit descendre. Si elle monte ou oscille, le pas $\\eta$ est trop grand.',
        draw(h) {
          const S = h.shared;
          const svg = h.svg(720, 420);
          const g = h.el('g', { transform: 'translate(8,70) scale(0.47)' }, svg);
          const v = buildNet(h, svg, { parent: g, bare: true });
          let net = clone(S.net0), hist = [S.f0.E], eta = ETA, speed = 10, logScale = true, running = false, tok = 0;
          const left = 8;
          h.el('text', { x: 160, y: 40, 'text-anchor': 'middle', class: 'muted', style: 'font-size:12px', text: 'Réseau (épaisseur \u221D |w|)' }, svg);
          const tIt = h.el('text', { x: left + 10, y: 255, class: 'mono', style: 'font-size:13px' }, svg);
          const tE = h.el('text', { x: left + 10, y: 277, class: 'mono', style: 'font-size:13px' }, svg);
          const tO = h.el('text', { x: left + 10, y: 299, class: 'mono', style: 'font-size:13px' }, svg);
          const cx0 = 400, cx1 = 700, cy0 = 40, cy1 = 280;
          h.el('rect', { x: cx0 - 44, y: cy0 - 18, width: cx1 - cx0 + 58, height: cy1 - cy0 + 56, rx: 10, fill: 'var(--surface-2)', opacity: 0.5 }, svg);
          const cg = h.el('g', {}, svg);

          /**
           * Redraws the error-versus-iteration chart.
           */
          const chart = () => {
            while (cg.firstChild) cg.removeChild(cg.firstChild);
            const it = hist.length - 1;
            const marks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];
            const xmax = marks.find((m) => m >= it) || marks[marks.length - 1];
            const sx = h.scale(0, xmax, cx0, cx1);
            const sy = logScale
              ? (e) => cy1 - ((Math.max(-6, Math.min(0, Math.log10(Math.max(e, 1e-12)))) + 6) / 6) * (cy1 - cy0)
              : (e) => cy1 - (Math.min(e, 0.3) / 0.3) * (cy1 - cy0);
            h.el('line', { x1: cx0, y1: cy1, x2: cx1, y2: cy1, stroke: 'var(--ink)' }, cg);
            h.el('line', { x1: cx0, y1: cy0, x2: cx0, y2: cy1, stroke: 'var(--ink)' }, cg);
            const yt = logScale ? [0, -2, -4, -6] : [0, 0.1, 0.2, 0.3];
            yt.forEach((t) => {
              const yy = logScale ? sy(Math.pow(10, t)) : sy(t);
              h.el('line', { x1: cx0, y1: yy, x2: cx1, y2: yy, stroke: 'var(--line)', 'stroke-dasharray': '3 4' }, cg);
              h.el('text', { x: cx0 - 6, y: yy + 4, 'text-anchor': 'end', class: 'mono muted', style: 'font-size:11px', text: logScale ? '10' + sup(t) : h.fmt(t, 1) }, cg);
            });
            [0, 0.5, 1].forEach((t) => {
              h.el('text', { x: sx(xmax * t), y: cy1 + 16, 'text-anchor': 'middle', class: 'mono muted', style: 'font-size:11px', text: String(Math.round(xmax * t)) }, cg);
            });
            h.el('text', { x: cx1, y: cy1 + 34, 'text-anchor': 'end', class: 'muted', style: 'font-size:12px', text: 'itérations' }, cg);
            h.el('text', { x: cx0 - 40, y: cy0 - 4, class: 'muted', style: 'font-size:12px', text: 'erreur E' + (logScale ? ' (log)' : '') }, cg);
            const stride = Math.max(1, Math.ceil(hist.length / 300));
            const pts = [];
            for (let i = 0; i < hist.length; i += stride) pts.push(sx(i).toFixed(1) + ',' + sy(hist[i]).toFixed(1));
            pts.push(sx(it).toFixed(1) + ',' + sy(hist[it]).toFixed(1));
            h.el('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, cg);
            h.el('circle', { cx: sx(it), cy: sy(hist[it]), r: 4.5, fill: 'var(--accent)' }, cg);
          };

          /**
           * Refreshes the network drawing and the numeric read-outs.
           */
          const refresh = () => {
            const f = forward(net);
            v.paint(net);
            tIt.textContent = 'itération : ' + (hist.length - 1);
            tE.textContent = 'E = ' + h.fmt(f.E, 5);
            tO.textContent = 'o = ' + h.fmt(f.o, 3) + '  (cible ' + h.fmt(Y, 2) + ')';
            chart();
          };

          /**
           * Performs one full forward/backward/update iteration.
           */
          const iterate = () => {
            const f = forward(net);
            const gr = backward(net, f);
            net = applyStep(net, gr, eta);
            hist.push(forward(net).E);
          };

          let btn;
          /**
           * Starts or stops the automatic training loop.
           */
          const toggleRun = () => {
            running = !running;
            btn.innerHTML = running ? 'Pause' : 'Itérer';
            if (!running) return;
            const my = ++tok;
            h.loop(() => {
              if (!running || my !== tok) return false;
              for (let i = 0; i < speed; i++) iterate();
              refresh();
            });
          };
          btn = h.button('Itérer', toggleRun, 'primary');
          h.button('1 pas', () => { iterate(); refresh(); });
          h.button('Réinitialiser', () => {
            net = clone(S.net0);
            hist = [S.f0.E];
            refresh();
          });
          h.toggle([[1, '\u00D71'], [10, '\u00D710'], [100, '\u00D7100']], 10, (val) => { speed = val; });
          h.toggle([['log', 'E : log'], ['lin', 'E : linéaire']], 'log', (val) => { logScale = val === 'log'; chart(); });
          h.slider({ label: '$\\eta$', min: 0.05, max: 5, step: 0.05, value: ETA, decimals: 2, onInput(val) { eta = val; } });
          refresh();
          h.after(600, () => { if (!running) toggleRun(); });
        }
      },

      {
        title: 'Gradient qui s\'évanouit',
        duration: 14000,
        text: String.raw`<p>Dans un réseau profond, la rétropropagation multiplie <em>une dérivée par couche traversée</em>. Pour la sigmoïde, $g'(a)=g(a)(1-g(a))\le 0{,}25$ : le signal d'erreur est donc multiplié par au plus $0{,}25\cdot|w|$ à chaque couche.</p>
$$\frac{\partial E}{\partial w^{(L-l)}}\ \propto\ \prod_{m=1}^{l} w\,g'(a_m)\ \le\ (0{,}25\,|w|)^{l}$$
<p>Avec $|w|=1$, après 10 couches il reste $\approx 10^{-6}$ : les premières couches n'apprennent presque plus, c'est le <strong>vanishing gradient</strong> (évoqué page 20). Avec la ReLU, $g'=1$ pour $a>0$ : le produit ne rétrécit pas.</p>
<p>Barres rouges = sigmoïde, vertes = ReLU. Jouez avec le poids $|w|$ et la pré-activation $|a|$ (plus on s'éloigne de 0, plus la sigmoïde sature et plus $g'$ est petit).</p>`,
        note: 'C\'est une des raisons du succès de la <strong>ReLU</strong> et des variantes dans les réseaux profonds. Son défaut : si $a<0$ la dérivée est nulle (« neurone mort »).',
        check: {
          q: 'Quelle est la valeur maximale de la dérivée de la sigmoïde ?',
          choices: ['1', '0,5', '0,25'],
          answer: 2,
          explain: '$g\'(a)=g(a)(1-g(a))$ est maximale pour $g=0{,}5$ (i.e. $a=0$) et vaut $0{,}5\\times0{,}5=0{,}25$.'
        },
        draw(h) {
          const svg = h.svg(720, 400);
          const L = 10, X0 = 80, X1 = 700, TOP = 44, BASE = 290, H = BASE - TOP, slot = (X1 - X0) / L;
          let wAbs = 1, aPre = 0, mode = 'log';
          const gg = h.el('g', {}, svg);
          const bg = h.el('g', {}, svg);
          h.el('rect', { x: X0, y: 10, width: 12, height: 12, rx: 2, fill: 'var(--neg)' }, svg);
          h.el('text', { x: X0 + 18, y: 21, text: 'sigmoïde', style: 'font-size:12px' }, svg);
          h.el('rect', { x: X0 + 100, y: 10, width: 12, height: 12, rx: 2, fill: 'var(--ok)' }, svg);
          h.el('text', { x: X0 + 118, y: 21, text: 'ReLU', style: 'font-size:12px' }, svg);
          h.el('text', { x: X1, y: 21, 'text-anchor': 'end', class: 'muted', style: 'font-size:12px', text: 'intensité du gradient après l couches' }, svg);
          h.el('text', { x: 72, y: 330, 'text-anchor': 'end', class: 'muted', style: 'font-size:11px', text: 'couches' }, svg);
          h.el('text', { x: 72, y: 348, 'text-anchor': 'end', class: 'mono', style: 'font-size:11px;fill:var(--neg)', text: 'sigm.' }, svg);
          h.el('text', { x: 72, y: 364, 'text-anchor': 'end', class: 'mono', style: 'font-size:11px;fill:var(--ok)', text: 'ReLU' }, svg);
          h.el('text', { x: (X0 + X1) / 2, y: 388, 'text-anchor': 'middle', class: 'muted', style: 'font-size:12px', text: 'nombre de couches traversées en remontant depuis la sortie \u2192' }, svg);
          const bars = [], rowS = [], rowR = [];
          for (let l = 1; l <= L; l++) {
            const cx = X0 + slot * (l - 0.5);
            bars.push([
              h.el('rect', { x: cx - 22, width: 20, rx: 2, fill: 'var(--neg)' }, bg),
              h.el('rect', { x: cx + 2, width: 20, rx: 2, fill: 'var(--ok)' }, bg)
            ]);
            h.el('text', { x: cx, y: 312, 'text-anchor': 'middle', class: 'mono', style: 'font-size:12px', text: String(l) }, svg);
            rowS.push(h.el('text', { x: cx, y: 348, 'text-anchor': 'middle', class: 'mono', style: 'font-size:10.5px;fill:var(--neg)' }, svg));
            rowR.push(h.el('text', { x: cx, y: 364, 'text-anchor': 'middle', class: 'mono', style: 'font-size:10.5px;fill:var(--ok)' }, svg));
          }
          const out = h.readout('');

          /**
           * Maps a gradient magnitude to a height in pixels for the current scale.
           */
          const hOf = (val) => {
            if (mode === 'log') {
              const t = (Math.log10(Math.max(val, 1e-12)) + 7) / 10;
              return Math.max(1, Math.min(1, t) * H);
            }
            return Math.max(1, Math.min(1, val / 1.2) * H);
          };

          /**
           * Redraws grid and bars; p is the growth factor (0..1) used for the intro animation.
           */
          const render = (p) => {
            while (gg.firstChild) gg.removeChild(gg.firstChild);
            const ticks = mode === 'log' ? [-6, -4, -2, 0, 2] : [0, 0.25, 0.5, 0.75, 1];
            ticks.forEach((t) => {
              const val = mode === 'log' ? Math.pow(10, t) : t;
              const yy = mode === 'log' ? BASE - hOf(val) : BASE - (t / 1.2) * H;
              h.el('line', { x1: X0, y1: yy, x2: X1, y2: yy, stroke: t === 0 && mode === 'log' ? 'var(--muted)' : 'var(--line)', 'stroke-dasharray': t === 0 && mode === 'log' ? '' : '3 4' }, gg);
              h.el('text', { x: X0 - 6, y: yy + 4, 'text-anchor': 'end', class: 'mono muted', style: 'font-size:11px', text: mode === 'log' ? '10' + sup(t) : h.fmt(t, 2) }, gg);
            });
            h.el('line', { x1: X0, y1: BASE, x2: X1, y2: BASE, stroke: 'var(--ink)' }, gg);
            const s = sig(aPre), gp = s * (1 - s);
            const qs = wAbs * gp, qr = wAbs;
            for (let l = 1; l <= L; l++) {
              const vs = Math.pow(qs, l), vr = Math.pow(qr, l);
              const hs = hOf(vs) * p, hr = hOf(vr) * p;
              h.attr(bars[l - 1][0], { y: BASE - hs, height: hs });
              h.attr(bars[l - 1][1], { y: BASE - hr, height: hr });
              rowS[l - 1].textContent = sci(vs);
              rowR[l - 1].textContent = sci(vr);
            }
            out('$g\'(a)=' + tn(gp, 3) + '$ \u2192 facteur sigmoïde par couche : ' + h.fmt(qs, 3) + ' \u2022 après 10 couches : ' + sci(Math.pow(qs, L)) + ' (ReLU : ' + sci(Math.pow(qr, L)) + ')');
          };
          render(0);
          h.tween(1200, (p) => render(p));
          h.slider({ label: '$|w|$', min: 0.5, max: 2, step: 0.1, value: 1, decimals: 1, onInput(val) { wAbs = val; render(1); } });
          h.slider({ label: '$|a|$', min: 0, max: 4, step: 0.1, value: 0, decimals: 1, onInput(val) { aPre = val; render(1); } });
          h.toggle([['log', 'échelle log'], ['lin', 'linéaire']], 'log', (val) => { mode = val; render(1); });
        }
      }
    ]
  });
})();
