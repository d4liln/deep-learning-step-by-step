/**
 * Lesson: model averaging (ensembles, bagging, boosting) and dropout as implicit averaging.
 */
(function () {
  /**
   * Seeded generator with a few discarded values so that neighbouring seeds decorrelate.
   */
  function makeRng(h, seed) {
    const r = h.rng(seed * 7919 + 13);
    for (let i = 0; i < 5; i++) r();
    return r;
  }

  /**
   * Removes every child of a DOM node.
   */
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  /**
   * Converts a pointer event into SVG viewBox coordinates.
   */
  function svgPoint(svg, ev) {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  /**
   * Builds an SVG path string by sampling fn on [a, b].
   */
  function curvePath(fn, a, b, n, px, py, clampLo, clampHi) {
    let d = '';
    for (let i = 0; i <= n; i++) {
      const x = a + (b - a) * i / n;
      const y = Math.max(clampLo, Math.min(clampHi, fn(x)));
      d += (i ? 'L' : 'M') + px(x).toFixed(1) + ',' + py(y).toFixed(1);
    }
    return d;
  }

  /**
   * True boundary of the toy problem.
   */
  function trueBoundary(x) {
    return 0.5 * Math.sin(1.6 * x);
  }

  /**
   * Builds the toy 2D dataset and five imperfect classifier boundaries (seeded).
   */
  function buildEnsemble(h) {
    const r = makeRng(h, 1);
    const pts = [];
    for (let i = 0; i < 46; i++) {
      const x = -2.8 + 5.6 * r(), y = (r() - 0.5) * 3;
      let lab = y > trueBoundary(x) ? 1 : -1;
      if (r() < 0.08) lab = -lab;
      pts.push({ x, y, lab });
    }
    const curves = [];
    for (let k = 0; k < 5; k++) {
      const a1 = 0.25 + 0.3 * r(), f1 = 0.8 + 2.5 * r(), p1 = 6.28 * r();
      const a2 = 0.2 + 0.25 * r(), f2 = 0.8 + 3 * r(), p2 = 6.28 * r();
      curves.push((x) => trueBoundary(x) + a1 * Math.sin(f1 * x + p1) + a2 * Math.sin(f2 * x + p2));
    }
    const median = (x) => curves.map((c) => c(x)).sort((a, b) => a - b)[2];
    return { pts, curves, median };
  }

  /**
   * Draws a tiny feed-forward network sketch with random signed edges.
   */
  function miniNet(h, svg, x0, y0, w, hgt, nHidden, seed) {
    const r = makeRng(h, seed);
    const shown = Math.min(nHidden, 7);
    const xs = [x0 + 22, x0 + w / 2, x0 + w - 22];
    const cy = y0 + hgt / 2;
    const col = (n, sp) => Array.from({ length: n }, (_, i) => cy + (i - (n - 1) / 2) * sp);
    const ins = col(2, 26), hid = col(shown, Math.min(20, (hgt - 20) / Math.max(1, shown - 1))), outs = col(1, 0);
    const edge = (xa, ya, xb, yb) => {
      const v = r() * 2 - 1;
      h.el('line', { x1: xa, y1: ya, x2: xb, y2: yb, stroke: v >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': 0.4 + 1.8 * Math.abs(v), 'stroke-opacity': 0.8 }, svg);
    };
    ins.forEach((ya) => hid.forEach((yb) => edge(xs[0], ya, xs[1], yb)));
    hid.forEach((ya) => outs.forEach((yb) => edge(xs[1], ya, xs[2], yb)));
    ins.forEach((y) => h.el('circle', { cx: xs[0], cy: y, r: 7, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.2 }, svg));
    hid.forEach((y) => h.el('circle', { cx: xs[1], cy: y, r: 7, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.2 }, svg));
    outs.forEach((y) => h.el('circle', { cx: xs[2], cy: y, r: 7, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.2 }, svg));
  }

  /**
   * Layout of the dropout network: layer sizes and node coordinates.
   */
  function dropoutLayout() {
    const sizes = [4, 6, 6, 3], xs = [90, 260, 430, 600], sp = [62, 54, 54, 70];
    return sizes.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: xs[l], y: 215 + (i - (n - 1) / 2) * sp[l] })));
  }

  /**
   * Draws a random dropout mask for layers 0..2 (output layer is never dropped).
   */
  function dropMask(r, sizes, p) {
    return sizes.map((n, l) => {
      if (l === sizes.length - 1) return new Array(n).fill(true);
      const keep = Array.from({ length: n }, () => r() >= p);
      if (!keep.some(Boolean)) keep[Math.floor(r() * n)] = true;
      return keep;
    });
  }

  /**
   * Runs AdaBoost with decision stumps on a small seeded 2D dataset.
   */
  function buildBoost(h) {
    const r = makeRng(h, 9);
    const pts = [];
    for (let i = 0; i < 12; i++) {
      const x = 0.08 + 0.84 * r(), y = 0.08 + 0.84 * r();
      pts.push({ x, y, lab: (x + y > 1 && !(x < 0.3)) || (x > 0.7 && y < 0.4) ? 1 : -1 });
    }
    let w = pts.map(() => 1 / pts.length);
    const rounds = [];
    for (let t = 0; t < 4; t++) {
      let best = null;
      ['x', 'y'].forEach((f) => pts.forEach((p) => {
        const th = p[f] + 0.001;
        [1, -1].forEach((s) => {
          let e = 0;
          pts.forEach((q, i) => { if ((q[f] > th ? s : -s) !== q.lab) e += w[i]; });
          if (!best || e < best.e - 1e-9) best = { f, th, s, e };
        });
      }));
      const alpha = 0.5 * Math.log((1 - best.e) / Math.max(best.e, 1e-9));
      const mis = pts.map((q) => (q[best.f] > best.th ? best.s : -best.s) !== q.lab);
      let nw = w.map((v, i) => v * Math.exp(mis[i] ? alpha : -alpha));
      const z = nw.reduce((a, b) => a + b, 0);
      nw = nw.map((v) => v / z);
      rounds.push({ f: best.f, th: best.th, s: best.s, e: best.e, alpha, mis, w: w.slice(), nw });
      w = nw;
    }
    return { pts, rounds };
  }

  Course.register({
    id: 'regularisation',
    order: 8,
    title: 'Moyennage de modèles & Dropout',
    summary: 'Combiner plusieurs classifieurs (vote, bagging, boosting) améliore les performances ; le <em>dropout</em> en est une version bon marché pour les réseaux de neurones.',
    pdfPages: '30-32',
    init(shared) {
      shared.dropP = 0.5;
    },
    steps: [
      {
        title: 'Moyennage de modèles',
        duration: 12000,
        text: '<p>Idée : au lieu d\'un seul classifieur, on en apprend <strong>plusieurs, différents, pour le même problème</strong>, puis on les <strong>combine</strong>. C\'est une <em>méthode ensembliste</em>.</p>' +
          '<p>Combinaisons possibles : <strong>vote</strong> (éventuellement pondéré) ou <strong>agrégation de scores</strong> :</p>' +
          '$$\\hat{y}(x)=\\arg\\max_c \\sum_{k=1}^{K} w_k\\,p_k(c\\mid x)$$' +
          '<p>Chaque classifieur se trompe à des endroits différents ; leur vote majoritaire a une frontière plus lisse. Cliquez dans le plan pour placer un point de test.</p>',
        note: 'Les erreurs « indépendantes » des classifieurs se compensent dans le vote : l\'ensemble est souvent meilleur que chacun de ses membres.',
        check: {
          q: 'Pourquoi un vote majoritaire peut-il être meilleur que chaque classifieur pris seul ?',
          choices: [
            'Parce que les classifieurs se trompent à des endroits différents et que leurs erreurs se compensent',
            'Parce que le vote ajoute des paramètres au modèle',
            'Parce que tous les classifieurs sont identiques'
          ],
          answer: 0,
          explain: 'Si les classifieurs sont diversifiés, une erreur isolée est « noyée » par la majorité des autres.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const E = buildEnsemble(h);
          const px = h.scale(-3, 3, 30, 500), py = h.scale(-2, 2, 400, 20);
          h.el('rect', { x: 30, y: 20, width: 470, height: 380, fill: 'var(--surface-2)', 'fill-opacity': 0.5, stroke: 'var(--line)' }, svg);
          const indiv = h.el('g', {}, svg);
          E.curves.forEach((c) => h.el('path', { d: curvePath(c, -3, 3, 90, px, py, -2, 2), fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.6, 'stroke-opacity': 0.85 }, indiv));
          h.el('path', { d: curvePath(E.median, -3, 3, 90, px, py, -2, 2), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 4, 'stroke-linecap': 'round' }, svg);
          E.pts.forEach((p) => h.el('circle', { cx: px(p.x), cy: py(p.y), r: 4.5, fill: p.lab > 0 ? 'var(--pos)' : 'var(--neg)', stroke: 'var(--ink)', 'stroke-width': 0.8 }, svg));
          const acc = (f) => E.pts.filter((p) => (p.y > f(p.x) ? 1 : -1) === p.lab).length / E.pts.length;
          const accs = E.curves.map(acc);
          const ensAcc = acc(E.median);
          const test = h.el('g', {}, svg);
          const hit = h.el('rect', { x: 30, y: 20, width: 470, height: 380, fill: 'transparent', style: 'cursor:crosshair' }, svg);
          const panel = h.el('g', {}, svg);
          h.el('text', { x: 610, y: 38, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: 'var(--ink)', text: 'Votes' }, svg);
          let token = 0, tx = -0.6, ty = 0.9;
          /**
           * Places the test point and animates the five votes one by one.
           */
          const castVotes = async () => {
            const my = ++token;
            clear(test);
            clear(panel);
            h.el('circle', { cx: px(tx), cy: py(ty), r: 9, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 3 }, test);
            h.el('text', { x: px(tx), y: py(ty) + 4, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: 'var(--ink)', text: '?' }, test);
            let a = 0, b = 0;
            for (let k = 0; k < 5; k++) {
              await h.sleep(420);
              if (my !== token) return;
              const v = ty > E.curves[k](tx) ? 1 : -1;
              if (v > 0) a++; else b++;
              const y = 78 + k * 46;
              h.el('text', { x: 548, y: y + 5, 'font-size': 14, fill: 'var(--ink)', text: 'C' + (k + 1) }, panel);
              h.el('circle', { cx: 620, cy: y, r: 14, fill: v > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.85 }, panel);
              h.el('text', { x: 620, y: y + 5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: 'var(--surface)', text: v > 0 ? 'A' : 'B' }, panel);
              h.el('text', { x: 665, y: y + 5, 'text-anchor': 'middle', 'font-size': 11.5, class: 'mono', fill: 'var(--muted)', text: 'acc ' + Math.round(accs[k] * 100) + '%' }, panel);
            }
            await h.sleep(450);
            if (my !== token) return;
            const win = a > b ? 'A' : 'B';
            h.el('rect', { x: 540, y: 300, width: 160, height: 64, rx: 10, fill: 'var(--accent)', 'fill-opacity': 0.15, stroke: 'var(--accent)', 'stroke-width': 2 }, panel);
            h.el('text', { x: 620, y: 325, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--ink)', text: 'Vote majoritaire : ' + Math.max(a, b) + ' contre ' + Math.min(a, b) }, panel);
            h.el('text', { x: 620, y: 352, 'text-anchor': 'middle', 'font-size': 17, 'font-weight': 700, fill: 'var(--accent)', text: 'Classe ' + win }, panel);
          };
          hit.addEventListener('click', (ev) => {
            const p = svgPoint(svg, ev);
            tx = px.invert(p.x);
            ty = py.invert(p.y);
            castVotes();
          });
          h.toggle([['both', 'Individuels + vote'], ['only', 'Vote seul']], 'both', (v) => { indiv.style.opacity = v === 'both' ? 1 : 0; });
          h.readout('Précision (apprentissage) : classifieurs ' + Math.round(Math.min(...accs) * 100) + '–' + Math.round(Math.max(...accs) * 100) + '&nbsp;%, <b>vote ' + Math.round(ensAcc * 100) + '&nbsp;%</b>');
          h.el('text', { x: 40, y: 414, 'font-size': 12, fill: 'var(--muted)', text: 'Gris : frontières des 5 classifieurs · Bleu : frontière du vote majoritaire' }, svg);
          castVotes();
        }
      },
      {
        title: 'Init & complexité',
        duration: 11000,
        text: '<p>Combiner des classifieurs <em>identiques</em> ne sert à rien : il faut de la <strong>diversité</strong>. Deux leviers simples :</p>' +
          '<ul><li><strong>plusieurs initialisations</strong> : même architecture, mais l\'apprentissage aboutit à des minima différents ;</li><li><strong>complexité variable</strong> : un nombre différent de neurones sur la couche cachée.</li></ul>' +
          '<p>Trop peu de neurones : sous-apprentissage. Trop : sur-apprentissage (frontière qui épouse le bruit). L\'ensemble moyenne ces défauts.</p>',
        note: 'Plus les erreurs des classifieurs sont décorrélées, plus le moyennage est efficace.',
        draw(h) {
          const svg = h.svg(720, 420);
          const E = buildEnsemble(h);
          let mode = 'init';
          const g = h.el('g', {}, svg);
          /**
           * Redraws the three panels for the selected diversity source.
           */
          const render = () => {
            clear(g);
            const hs = mode === 'init' ? [5, 5, 5] : [1, 5, 20];
            const caps = mode === 'init' ? ['Initialisation 1', 'Initialisation 2', 'Initialisation 3'] : ['Sous-apprentissage', 'Bon compromis', 'Sur-apprentissage'];
            const fns = mode === 'init'
              ? [0.25, 0.45, 0.35].map((a, i) => (x) => trueBoundary(x) + a * Math.sin(1.4 * x + i * 2.1 + 0.5))
              : [(x) => 0.3 * x, (x) => trueBoundary(x) + 0.1 * Math.sin(2.2 * x), (x) => trueBoundary(x) + 0.55 * Math.sin(7 * x + 0.4)];
            for (let i = 0; i < 3; i++) {
              const x0 = 20 + i * 235;
              h.el('rect', { x: x0, y: 14, width: 220, height: 392, rx: 10, fill: 'var(--surface-2)', 'fill-opacity': 0.5, stroke: 'var(--line)' }, g);
              miniNet(h, g, x0, 22, 220, 130, hs[i], 30 + i + (mode === 'init' ? 0 : 7));
              h.el('text', { x: x0 + 110, y: 168, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--ink)', text: hs[i] + ' neurone' + (hs[i] > 1 ? 's' : '') + ' cachés' }, g);
              const px = h.scale(-3, 3, x0 + 10, x0 + 210), py = h.scale(-2, 2, 380, 190);
              h.el('rect', { x: x0 + 10, y: 190, width: 200, height: 190, fill: 'var(--surface)', stroke: 'var(--line)' }, g);
              h.el('path', { d: curvePath(fns[i], -3, 3, 120, px, py, -2, 2), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.8 }, g);
              E.pts.forEach((p) => h.el('circle', { cx: px(p.x), cy: py(p.y), r: 2.8, fill: p.lab > 0 ? 'var(--pos)' : 'var(--neg)' }, g));
              h.el('text', { x: x0 + 110, y: 398, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: 'var(--ink)', text: caps[i] }, g);
            }
          };
          h.toggle([['init', 'Plusieurs initialisations'], ['cx', 'Complexité variable']], mode, (v) => { mode = v; render(); });
          render();
        }
      },
      {
        title: 'Bagging',
        duration: 15000,
        text: '<p>Le <strong>bagging</strong> (<em>bootstrap aggregating</em>) fabrique de la diversité en changeant les <strong>données</strong> : chaque classifieur est appris sur un échantillon de la base obtenu par <strong>rééchantillonnage avec remise</strong>.</p>' +
          '<p>On tire $N$ exemples parmi $N$, un par un, en <em>remettant</em> chaque exemple tiré : certains apparaissent plusieurs fois, d\'autres jamais (environ $37\\,\\%$ en moyenne, soit $(1-1/N)^N\\to e^{-1}$).</p>',
        note: 'Chaque classifieur voit une base légèrement différente ; on moyenne ensuite leurs sorties (vote ou moyenne des scores).',
        check: {
          q: 'Dans un tirage bootstrap de $N$ exemples avec remise, un même exemple peut-il apparaître plusieurs fois ?',
          choices: ['Non, jamais', 'Oui, et d\'autres exemples peuvent ne pas être tirés du tout', 'Oui, mais tous les exemples sont toujours tirés'],
          answer: 1,
          explain: 'Avec remise, chaque tirage est indépendant : doublons et exemples absents sont normaux.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const N = 10, labs = [1, 1, -1, 1, -1, -1, 1, -1, 1, -1];
          const cx = (i) => 66 + i * 65;
          const Y0 = 120, Y1 = 275;
          h.el('text', { x: 360, y: 52, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Base d\'apprentissage (N = 10)' }, svg);
          h.el('text', { x: 360, y: 232, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Échantillon bootstrap : 10 tirages avec remise' }, svg);
          const orig = [];
          for (let i = 0; i < N; i++) {
            const g = h.el('g', { style: 'transition:opacity .5s' }, svg);
            h.el('circle', { cx: cx(i), cy: Y0, r: 20, fill: labs[i] > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.8, stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
            h.el('text', { x: cx(i), y: Y0 + 5, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: 'var(--surface)', text: String(i + 1) }, g);
            const badge = h.el('text', { x: cx(i), y: Y0 + 42, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, class: 'mono', fill: 'var(--accent-2)', text: '' }, svg);
            orig.push({ g, badge });
          }
          for (let s = 0; s < N; s++) {
            h.el('rect', { x: cx(s) - 22, y: Y1 - 22, width: 44, height: 44, rx: 8, fill: 'none', stroke: 'var(--line)', 'stroke-dasharray': '4 3' }, svg);
          }
          const dyn = h.el('g', {}, svg);
          const note = h.label(svg, 30, 322, 660, 90, '', 'center');
          let seed = 100, token = 0;
          /**
           * Draws a fresh bootstrap sample, one animated draw at a time.
           */
          const run = async () => {
            const my = ++token;
            const r = makeRng(h, seed);
            clear(dyn);
            const counts = new Array(N).fill(0);
            orig.forEach((o) => { o.badge.textContent = ''; o.g.style.opacity = 1; });
            note.innerHTML = 'Tirage en cours...';
            for (let s = 0; s < N; s++) {
              const idx = Math.floor(r() * N);
              const dup = counts[idx] > 0;
              counts[idx]++;
              h.attr(orig[idx].g.firstChild, { stroke: 'var(--accent)', 'stroke-width': 4 });
              const t = h.el('g', {}, dyn);
              h.el('circle', { cx: 0, cy: 0, r: 20, fill: labs[idx] > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.8, stroke: dup ? 'var(--warn)' : 'var(--ink)', 'stroke-width': dup ? 4 : 1.5 }, t);
              h.el('text', { x: 0, y: 5, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: 'var(--surface)', text: String(idx + 1) }, t);
              await h.tween(380, (p) => {
                h.attr(t, { transform: 'translate(' + h.lerp(cx(idx), cx(s), p) + ',' + h.lerp(Y0, Y1, p) + ')' });
              });
              if (my !== token) return;
              h.attr(orig[idx].g.firstChild, { stroke: 'var(--ink)', 'stroke-width': 1.5 });
              orig[idx].badge.textContent = counts[idx] > 1 ? '×' + counts[idx] : '×1';
              await h.sleep(90);
              if (my !== token) return;
            }
            const absent = counts.filter((c) => c === 0).length;
            const dups = counts.filter((c) => c > 1).length;
            orig.forEach((o, i) => { if (counts[i] === 0) o.g.style.opacity = 0.25; });
            note.innerHTML = '<b>' + absent + '</b> exemple' + (absent > 1 ? 's' : '') + ' jamais tiré' + (absent > 1 ? 's' : '') + ' (grisé' + (absent > 1 ? 's' : '') + ') · <b>' + dups + '</b> tiré' + (dups > 1 ? 's' : '') + ' plusieurs fois (contour orange).<br>Un classifieur est appris sur cet échantillon ; on répète pour obtenir $K$ classifieurs différents.';
            h.typeset(note);
          };
          h.button('Nouveau tirage', () => { seed++; run(); }, 'primary');
          run();
        }
      },
      {
        title: 'Boosting',
        duration: 20000,
        text: '<p>Le <strong>boosting</strong> apprend les classifieurs <strong>séquentiellement</strong> en se focalisant sur les exemples <em>mal classés</em> par les précédents.</p>' +
          '<p>Chaque exemple a un poids (rayon du disque). Après le tour $t$ : erreur pondérée $\\varepsilon_t$, poids du classifieur $\\alpha_t=\\tfrac12\\ln\\frac{1-\\varepsilon_t}{\\varepsilon_t}$, puis les exemples <strong>mal classés</strong> voient leur poids <strong>augmenter</strong>.</p>' +
          '$$H(x)=\\mathrm{sign}\\Big(\\sum_t \\alpha_t\\,h_t(x)\\Big)$$' +
          '<p>Les $\\alpha_t$ sont les <em>poids de combinaison optimaux</em> : un bon classifieur pèse plus dans le vote final.</p>',
        note: 'Ici, les classifieurs faibles sont de simples « stumps » (un seuil sur une coordonnée), et leur combinaison classe correctement les 12 points.',
        draw(h) {
          const svg = h.svg(720, 420);
          const B = buildBoost(h);
          const px = h.scale(0, 1, 50, 390), py = h.scale(0, 1, 380, 40);
          h.el('rect', { x: 30, y: 24, width: 380, height: 372, fill: 'var(--surface-2)', 'fill-opacity': 0.5, stroke: 'var(--line)' }, svg);
          const shade = h.el('g', {}, svg);
          const line = h.el('line', { x1: 0, y1: 0, x2: 0, y2: 0, stroke: 'var(--accent)', 'stroke-width': 3.5, 'stroke-opacity': 0 }, svg);
          const rad = (w) => 4 + 30 * Math.sqrt(w);
          const dots = B.pts.map((p) => {
            const g = h.el('g', {}, svg);
            const c = h.el('circle', { cx: px(p.x), cy: py(p.y), r: rad(1 / 12), fill: p.lab > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.75, stroke: 'var(--ink)', 'stroke-width': 1.2 }, g);
            return c;
          });
          h.el('text', { x: 560, y: 36, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: 'var(--ink)', text: 'Classifieurs faibles' }, svg);
          const list = h.el('g', {}, svg);
          const status = h.el('text', { x: 220, y: 414, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: 'Rayon du disque = poids de l\'exemple' }, svg);
          const maxAlpha = Math.max(...B.rounds.map((R) => R.alpha));
          let token = 0;
          /**
           * Sets disk radii from a weight vector.
           */
          const setRadii = (w) => dots.forEach((c, i) => h.attr(c, { r: rad(w[i]) }));
          /**
           * Draws the half-planes and the threshold line of a stump.
           */
          const showStump = (R) => {
            clear(shade);
            const v = R.f === 'x';
            const th = v ? px(R.th) : py(R.th);
            const regions = v ? [[30, 24, th - 30, 372, -R.s], [th, 24, 410 - th, 372, R.s]] : [[30, 24, 380, th - 24, R.s], [30, th, 380, 396 - th, -R.s]];
            regions.forEach(([x, y, w, hh, cls]) => h.el('rect', { x, y, width: Math.max(0, w), height: Math.max(0, hh), fill: cls > 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': 0.12 }, shade));
            h.attr(line, v ? { x1: th, y1: 24, x2: th, y2: 396, 'stroke-opacity': 1 } : { x1: 30, y1: th, x2: 410, y2: th, 'stroke-opacity': 1 });
          };
          /**
           * Plays the rounds one after another: stump, errors, then weight update.
           */
          const play = async () => {
            const my = ++token;
            clear(list);
            clear(shade);
            h.attr(line, { 'stroke-opacity': 0 });
            dots.forEach((c) => h.attr(c, { stroke: 'var(--ink)', 'stroke-width': 1.2 }));
            setRadii(B.pts.map(() => 1 / 12));
            status.textContent = 'Poids initiaux identiques : 1/12';
            await h.sleep(900);
            for (let t = 0; t < B.rounds.length; t++) {
              if (my !== token) return;
              const R = B.rounds[t];
              showStump(R);
              dots.forEach((c, i) => h.attr(c, R.mis[i] ? { stroke: 'var(--bad)', 'stroke-width': 4.5 } : { stroke: 'var(--ink)', 'stroke-width': 1.2 }));
              const y = 76 + t * 62;
              h.el('text', { x: 450, y: y, 'font-size': 14, fill: 'var(--ink)', 'font-weight': 700, text: 'Tour ' + (t + 1) }, list);
              h.el('text', { x: 450, y: y + 20, 'font-size': 12.5, class: 'mono', fill: 'var(--muted)', text: 'ε = ' + h.fmt(R.e, 2) + ' · α = ' + h.fmt(R.alpha, 2) }, list);
              h.el('rect', { x: 450, y: y + 28, width: Math.max(4, 230 * R.alpha / maxAlpha), height: 9, rx: 3, fill: 'var(--accent)' }, list);
              status.textContent = 'Tour ' + (t + 1) + ' : ' + R.mis.filter(Boolean).length + ' exemple(s) mal classé(s) (contour rouge)';
              await h.sleep(1500);
              if (my !== token) return;
              status.textContent = 'Mise à jour : les exemples mal classés grossissent';
              const from = R.w, to = R.nw;
              await h.tween(900, (p) => setRadii(from.map((v, i) => v + (to[i] - v) * p)));
            }
            if (my !== token) return;
            clear(shade);
            h.attr(line, { 'stroke-opacity': 0 });
            dots.forEach((c) => h.attr(c, { stroke: 'var(--ink)', 'stroke-width': 1.2 }));
            status.textContent = 'Vote final pondéré par les α : 12 / 12 exemples bien classés';
          };
          h.button('Rejouer le boosting', play, 'primary');
          play();
        }
      },
      {
        title: 'Dropout',
        duration: 20000,
        text: '<p>Entraîner et stocker des dizaines de réseaux coûte cher. Le <strong>dropout</strong> obtient un effet voisin avec un seul réseau.</p>' +
          '<p>À <strong>chaque itération</strong> (gradient stochastique), lors de la propagation avant, on <strong>inhibe</strong> (sortie = 0) chaque neurone de la couche d\'entrée et des couches cachées avec une probabilité $p$. On <strong>rétropropage l\'erreur normalement</strong> dans le réseau restant.</p>' +
          '<p>Un nouveau sous-réseau est tiré à chaque itération.</p>',
        note: 'Seuls les neurones actifs participent au calcul et sont mis à jour : les neurones éteints (pointillés) ne reçoivent pas de gradient à cette itération.',
        check: {
          q: 'Avec $p = 0{,}5$, quelle proportion des neurones cachés est inhibée en moyenne à une itération ?',
          choices: ['Aucun', 'La moitié', 'Tous'],
          answer: 1,
          explain: 'Chaque neurone est éteint indépendamment avec probabilité $p=0{,}5$ : en moyenne la moitié.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const L = dropoutLayout();
          const sizes = L.map((c) => c.length);
          const edges = [];
          for (let l = 0; l < L.length - 1; l++) {
            edges.push([]);
            L[l].forEach((a, i) => L[l + 1].forEach((b, j) => {
              const el = h.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-opacity': 0.4, style: 'transition:stroke-opacity .4s' }, svg);
              edges[l].push({ el, i, j });
            }));
          }
          const nodes = L.map((col) => col.map((n) => {
            const g = h.el('g', {}, svg);
            const c = h.el('circle', { cx: n.x, cy: n.y, r: 20, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, g);
            const x = h.el('text', { x: n.x, y: n.y + 6, 'text-anchor': 'middle', 'font-size': 18, 'font-weight': 700, fill: 'var(--bad)', text: '×', 'fill-opacity': 0 }, g);
            return { g, c, x };
          }));
          ['Entrée', 'Cachée 1', 'Cachée 2', 'Sortie'].forEach((t, l) => h.el('text', { x: L[l][0].x, y: 392, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: t }, svg));
          const phase = h.el('text', { x: 360, y: 34, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: 'var(--ink)', text: '' }, svg);
          const ro = h.readout('');
          let iter = 0, token = 0, auto = true, p = h.shared.dropP;
          /**
           * Applies the dropout mask of the current iteration to nodes and edges.
           */
          const applyMask = (mask) => {
            nodes.forEach((col, l) => col.forEach((n, i) => {
              const on = mask[l][i];
              n.g.style.transition = 'opacity .4s';
              n.g.style.opacity = on ? 1 : 0.3;
              h.attr(n.c, on ? { 'stroke-dasharray': '0', fill: 'var(--surface)' } : { 'stroke-dasharray': '4 3', fill: 'var(--surface-2)' });
              h.attr(n.x, { 'fill-opacity': on ? 0 : 1 });
            }));
            edges.forEach((layer, l) => layer.forEach((e) => {
              const on = mask[l][e.i] && mask[l + 1][e.j];
              h.attr(e.el, { stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-opacity': on ? 0.5 : 0.05 });
            }));
          };
          /**
           * Colours the active edges of one layer pair for the forward or backward pass.
           */
          const flash = (mask, l, color, on) => {
            edges[l].forEach((e) => {
              if (mask[l][e.i] && mask[l + 1][e.j]) h.attr(e.el, on ? { stroke: 'var(--' + color + ')', 'stroke-width': 2.4, 'stroke-opacity': 0.95 } : { stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-opacity': 0.5 });
            });
          };
          /**
           * Draws a new mask and plays forward then backward propagation through it.
           */
          const step = async () => {
            const my = ++token;
            iter++;
            const r = makeRng(h, iter * 131 + Math.round(p * 100) * 17 + 3);
            const mask = dropMask(r, sizes, p);
            const active = mask.reduce((s, col, l) => s + (l < 3 ? col.filter(Boolean).length : 0), 0);
            phase.textContent = 'Itération ' + iter + ' · tirage des neurones inhibés';
            ro('Neurones actifs : $' + active + ' / ' + (sizes[0] + sizes[1] + sizes[2]) + '$ (entrée + cachées) · $p=' + h.fmt(p, 2) + '$');
            applyMask(mask);
            await h.sleep(900);
            if (my !== token) return;
            phase.textContent = 'Itération ' + iter + ' · propagation avant';
            for (let l = 0; l < 3; l++) {
              flash(mask, l, 'accent', true);
              await h.sleep(420);
              if (my !== token) return;
            }
            phase.textContent = 'Itération ' + iter + ' · rétropropagation de l\'erreur';
            for (let l = 2; l >= 0; l--) {
              flash(mask, l, 'accent-2', true);
              await h.sleep(420);
              if (my !== token) return;
            }
            await h.sleep(300);
            if (my !== token) return;
            for (let l = 0; l < 3; l++) flash(mask, l, 'ink', false);
          };
          h.slider({ label: 'Probabilité $p$', min: 0, max: 0.9, step: 0.05, value: p, decimals: 2, onInput: (v) => { p = v; h.shared.dropP = v; iter--; step(); } });
          h.button('Itération suivante', step, 'primary');
          h.toggle([['on', 'Lecture auto'], ['off', 'Manuel']], 'on', (v) => { auto = v === 'on'; });
          h.every(4600, () => { if (auto) step(); });
          step();
        }
      },
      {
        title: 'Dropout au test',
        duration: 15000,
        text: '<p>Au <strong>test</strong>, on n\'inhibe plus rien : on utilise le réseau <strong>complet</strong>, mais on <strong>pondère les poids</strong> par la probabilité de garde $q=1-p$.</p>' +
          '<p>Pourquoi ? À l\'apprentissage, une entrée n\'est présente qu\'avec la probabilité $q$ : l\'entrée de la couche suivante vaut en moyenne $q\\sum_j w_jx_j$. Pour retrouver cette même valeur moyenne au test, on remplace $w_j$ par $q\\,w_j$.</p>' +
          '<p>En pratique, on peut aussi diviser par $q$ <em>pendant</em> l\'apprentissage (« inverted dropout ») et ne rien changer au test.</p>',
        note: 'La moyenne sur de nombreux masques est (pour la partie linéaire) égale à la sortie du réseau complet avec les poids multipliés par $q$.',
        check: {
          q: 'Avec $p = 0{,}2$ à l\'apprentissage, par quoi multiplie-t-on les poids au test ?',
          choices: ['$0{,}2$', '$0{,}8$', '$1{,}2$'],
          answer: 1,
          explain: 'On multiplie par la probabilité de garde $q = 1-p = 0{,}8$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const X = [1.0, 0.5, -1.0, 2.0, 1.5], Wt = [0.8, -0.5, 0.6, 0.4, -0.3];
          const inY = [80, 145, 210, 275, 340];
          const nx = 380, ny = 210;
          let p = h.shared.dropP, mode = 'train', seed = 1, token = 0;
          const dyn = h.el('g', {}, svg);
          const panel = h.el('g', {}, svg);
          const phase = h.el('text', { x: 250, y: 34, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 700, fill: 'var(--ink)', text: '' }, svg);
          /**
           * Computes the average pre-activation over many random masks.
           */
          const meanMasked = () => {
            const r = makeRng(h, 777);
            let s = 0;
            const M = 600;
            for (let m = 0; m < M; m++) X.forEach((x, j) => { if (r() >= p) s += Wt[j] * x; });
            return s / M;
          };
          /**
           * Redraws the neuron for the current mode, probability and mask seed.
           */
          const render = () => {
            clear(dyn);
            clear(panel);
            const q = 1 - p;
            const r = makeRng(h, seed * 31 + Math.round(p * 100));
            const keep = X.map(() => r() >= p);
            if (!keep.some(Boolean)) keep[2] = true;
            const full = X.reduce((s, x, j) => s + Wt[j] * x, 0);
            let a;
            if (mode === 'train') {
              a = X.reduce((s, x, j) => s + (keep[j] ? Wt[j] * x : 0), 0);
              phase.textContent = 'Apprentissage : entrées inhibées au hasard';
            } else {
              a = X.reduce((s, x, j) => s + q * Wt[j] * x, 0);
              phase.textContent = 'Test : toutes les entrées, poids × q';
            }
            X.forEach((x, j) => {
              const on = mode === 'test' || keep[j];
              const w = mode === 'test' ? q * Wt[j] : Wt[j];
              h.el('line', {
                x1: 150, y1: inY[j], x2: nx - 34, y2: ny,
                stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': 1 + 6 * Math.abs(w), 'stroke-opacity': on ? 0.85 : 0.08, 'stroke-linecap': 'round'
              }, dyn);
              h.el('circle', { cx: 120, cy: inY[j], r: 22, fill: on ? 'var(--surface)' : 'var(--surface-2)', stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-dasharray': on ? '0' : '4 3', 'stroke-opacity': on ? 1 : 0.4 }, dyn);
              h.el('text', { x: 120, y: inY[j] + 4, 'text-anchor': 'middle', 'font-size': 12.5, class: 'mono', fill: on ? 'var(--ink)' : 'var(--muted)', text: on ? h.fmt(x, 1) : '×' }, dyn);
              h.el('text', { x: 232, y: inY[j] + (ny - inY[j]) * 0.42 - 7, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: mode === 'test' ? 'var(--accent)' : 'var(--ink)', 'fill-opacity': on ? 1 : 0.2, text: (mode === 'test' ? 'q·' : '') + 'w=' + h.fmt(Wt[j], 1) + (mode === 'test' ? '=' + h.fmt(w, 2) : '') }, dyn);
            });
            h.el('circle', { cx: nx, cy: ny, r: 34, fill: 'var(--surface)', stroke: 'var(--accent)', 'stroke-width': 3 }, dyn);
            h.el('text', { x: nx, y: ny - 2, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Σ' }, dyn);
            h.el('text', { x: nx, y: ny + 16, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--accent)', text: h.fmt(a, 2) }, dyn);
            const mm = meanMasked();
            const rows = [
              ['Réseau complet, poids w', h.fmt(full, 2), 'var(--muted)'],
              mode === 'train' ? ['Ce masque', h.fmt(a, 2), 'var(--ink)'] : ['Test : poids q·w', h.fmt(a, 2), 'var(--accent)'],
              ['Moyenne de 600 masques', h.fmt(mm, 2), 'var(--accent-2)'],
              ['q × réseau complet', h.fmt(q * full, 2), 'var(--accent)']
            ];
            h.el('text', { x: 560, y: 120, 'font-size': 14, 'font-weight': 700, fill: 'var(--ink)', text: 'Entrée du neurone' }, panel);
            rows.forEach((row, i) => {
              h.el('text', { x: 560, y: 156 + i * 52, 'font-size': 12.5, fill: 'var(--muted)', text: row[0] }, panel);
              h.el('text', { x: 560, y: 178 + i * 52, 'font-size': 18, 'font-weight': 700, class: 'mono', fill: row[2], text: row[1] }, panel);
            });
          };
          h.toggle([['train', 'Apprentissage'], ['test', 'Test']], mode, (v) => { mode = v; render(); });
          h.button('Autre masque', () => { seed++; mode = 'train'; Array.from(h.controls.querySelectorAll('.ctl-toggle button')).forEach((b, i) => b.setAttribute('aria-pressed', String(i === 0))); render(); });
          h.slider({ label: 'Probabilité $p$', min: 0.1, max: 0.9, step: 0.1, value: Math.max(0.1, p), decimals: 1, onInput: (v) => { p = v; h.shared.dropP = v; render(); } });
          render();
        }
      },
      {
        title: 'Un ensemble qui partage',
        duration: 13000,
        text: '<p>Avec $n$ neurones inhibables, il existe $2^n$ sous-réseaux possibles. Chaque itération en entraîne un, tiré au hasard ; tous <strong>partagent les mêmes paramètres</strong>.</p>' +
          '<p>Le dropout revient donc à <strong>apprendre de nombreux modèles partageant des paramètres et à les moyenner</strong> : c\'est du moyennage de modèles, sans avoir à stocker ni entraîner $2^n$ réseaux séparés.</p>' +
          '<p>Le réseau complet aux poids pondérés approxime la moyenne de ces sous-réseaux au moment du test.</p>',
        note: 'Chaque neurone ne peut pas compter sur la présence de ses voisins : il doit apprendre des caractéristiques utiles seules (moins de co-adaptation). C\'est ce qui régularise.',
        draw(h) {
          const svg = h.svg(720, 420);
          const cols = [3, 4, 3];
          const cxs = [28, 82, 136];
          const gridG = h.el('g', {}, svg);
          const cells = [];
          for (let k = 0; k < 8; k++) {
            const gx = 20 + (k % 4) * 170, gy = 18 + Math.floor(k / 4) * 130;
            const g = h.el('g', { transform: 'translate(' + gx + ',' + gy + ')' }, gridG);
            h.el('rect', { x: 0, y: 0, width: 164, height: 118, rx: 8, fill: 'var(--surface-2)', 'fill-opacity': 0.5, stroke: 'var(--line)' }, g);
            const pos = cols.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: cxs[l] + 6, y: 59 + (i - (n - 1) / 2) * 24 })));
            const ed = [];
            for (let l = 0; l < 2; l++) pos[l].forEach((a, i) => pos[l + 1].forEach((b, j) => ed.push({ l, i, j, el: h.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'var(--ink)', 'stroke-opacity': 0.4 }, g) })));
            const nd = pos.map((col) => col.map((n) => h.el('circle', { cx: n.x, cy: n.y, r: 8, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.3 }, g)));
            cells.push({ ed, nd });
          }
          /**
           * Gives every miniature network its own random mask.
           */
          const shuffle = (round) => {
            cells.forEach((c, k) => {
              const r = makeRng(h, round * 17 + k + 5);
              const mask = dropMask(r, [3, 4, 3], 0.5);
              c.nd.forEach((col, l) => col.forEach((el, i) => h.attr(el, mask[l][i] ? { 'fill-opacity': 1, 'stroke-opacity': 1, 'stroke-dasharray': '0' } : { 'fill-opacity': 0.1, 'stroke-opacity': 0.35, 'stroke-dasharray': '3 2' })));
              c.ed.forEach((e) => h.attr(e.el, { 'stroke-opacity': mask[e.l][e.i] && mask[e.l + 1][e.j] ? 0.55 : 0.05 }));
            });
          };
          h.el('text', { x: 360, y: 292, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--muted)', text: 'Huit des sous-réseaux possibles — mêmes paramètres, masques différents' }, svg);
          h.el('line', { x1: 360, y1: 304, x2: 360, y2: 330, stroke: 'var(--accent)', 'stroke-width': 3, 'marker-end': 'url(#arrow-accent)' }, svg);
          h.el('text', { x: 380, y: 322, 'font-size': 13, fill: 'var(--accent)', text: 'moyenne' }, svg);
          const result = h.label(svg, 110, 338, 500, 70, '', 'center');
          let round = 0;
          h.every(1500, () => { round++; shuffle(round); });
          shuffle(0);
          const sl = h.slider({
            label: 'Neurones $n$', min: 4, max: 400, step: 1, value: 100, decimals: 0,
            onInput: (v) => { show(v); }
          });
          /**
           * Writes the number of sub-networks 2^n in the result panel.
           */
          const show = (n) => {
            const s = (2n ** BigInt(n)).toString();
            let txt;
            if (s.length <= 12) txt = s.replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
            else txt = '≈ ' + s[0] + ',' + s.slice(1, 3) + ' × 10<sup>' + (s.length - 1) + '</sup>';
            result.innerHTML = 'Avec $n=' + n + '$ neurones : <b style="color:var(--accent);font-size:1.25em">$2^{n}$ = ' + txt + '</b><br>sous-réseaux possibles';
            h.typeset(result);
          };
          show(sl.get());
        }
      },
      {
        title: 'Moins de sur-apprentissage',
        duration: 15000,
        text: '<p>Le dropout est une <strong>régularisation</strong> : il ajoute du bruit à l\'apprentissage. Le réseau ne peut plus mémoriser le bruit de la base d\'apprentissage.</p>' +
          '<p>Observez les courbes de perte :</p>' +
          '<ul><li><span style="color:var(--bad)">sans dropout</span> : la perte d\'apprentissage tombe vers $0$, mais la perte de validation remonte (<strong>sur-apprentissage</strong>) ;</li><li><span style="color:var(--ok)">avec dropout</span> : l\'apprentissage est plus lent et moins bas, mais la validation reste meilleure.</li></ul>' +
          '<p>Courbes schématiques, à titre d\'illustration.</p>',
        note: 'Ce qui compte est la perte de validation : le dropout réduit l\'écart entre apprentissage et validation.',
        check: {
          q: 'Que fait généralement le dropout à la perte d\'apprentissage et à la perte de validation (à la fin de l\'entraînement) ?',
          choices: [
            'Il augmente un peu la perte d\'apprentissage et diminue la perte de validation',
            'Il diminue les deux pertes',
            'Il augmente la perte de validation'
          ],
          answer: 0,
          explain: 'Le modèle ajuste moins finement le bruit d\'apprentissage (perte d\'apprentissage plus haute) mais généralise mieux (validation plus basse).'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const X0 = 70, X1 = 680, Y0 = 350, Y1 = 40, T = 100, VMAX = 1.4;
          const px = h.scale(0, T, X0, X1), py = h.scale(0, VMAX, Y0, Y1);
          h.el('line', { x1: X0, y1: Y0, x2: X1 + 6, y2: Y0, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
          h.el('line', { x1: X0, y1: Y0, x2: X0, y2: Y1 - 8, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
          [0, 0.4, 0.8, 1.2].forEach((v) => {
            h.el('line', { x1: X0, y1: py(v), x2: X1, y2: py(v), stroke: 'var(--line)', 'stroke-dasharray': '3 4' }, svg);
            h.el('text', { x: X0 - 8, y: py(v) + 4, 'text-anchor': 'end', 'font-size': 12, class: 'mono', fill: 'var(--muted)', text: h.fmt(v, 1) }, svg);
          });
          h.el('text', { x: (X0 + X1) / 2, y: 392, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: 'Époques d\'entraînement' }, svg);
          h.el('text', { x: 22, y: (Y0 + Y1) / 2, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', transform: 'rotate(-90 22 ' + (Y0 + Y1) / 2 + ')', text: 'Perte' }, svg);
          const f = {
            trNo: (t) => 0.02 + 1.1 * Math.exp(-t / 16),
            vaNo: (t) => 0.4 + 0.9 * Math.exp(-t / 14) + 0.006 * t,
            trDr: (t, s) => 0.02 + s * 0.25 * (1 - Math.exp(-t / 20)) + 1.1 * Math.exp(-t / (16 + 10 * s)),
            vaDr: (t, s) => 0.4 - 0.1 * s + 0.9 * Math.exp(-t / (14 + 8 * s)) + 0.006 * (1 - 0.85 * s) * t
          };
          const defs = [
            ['trNo', 'var(--bad)', '6 4', 'Apprentissage, sans dropout'],
            ['vaNo', 'var(--bad)', '0', 'Validation, sans dropout'],
            ['trDr', 'var(--ok)', '6 4', 'Apprentissage, avec dropout'],
            ['vaDr', 'var(--ok)', '0', 'Validation, avec dropout']
          ];
          const paths = defs.map(([k, col, dash]) => h.el('path', { d: '', fill: 'none', stroke: col, 'stroke-width': 3, 'stroke-dasharray': dash, 'stroke-linecap': 'round' }, svg));
          defs.forEach(([k, col, dash], i) => h.el('line', { x1: 90 + i * 150, y1: 24, x2: 90 + i * 150 + 28, y2: 24, stroke: col, 'stroke-width': 3, 'stroke-dasharray': dash }, svg));
          const leg = [['Appr. sans', 0], ['Valid. sans', 1], ['Appr. avec', 2], ['Valid. avec', 3]];
          leg.forEach(([name, i]) => h.el('text', { x: 124 + i * 150, y: 28, 'font-size': 12.5, fill: 'var(--ink)', text: name }, svg));
          const gap = h.el('g', { style: 'opacity:0;transition:opacity .5s' }, svg);
          const gapLine = h.el('line', { x1: 0, y1: 0, x2: 0, y2: 0, stroke: 'var(--warn)', 'stroke-width': 3 }, gap);
          const gapTxt = h.el('text', { x: 0, y: 0, 'font-size': 12.5, 'font-weight': 700, fill: 'var(--warn)', text: 'écart appr./valid.' }, gap);
          let s = h.shared.dropP / 0.8, token = 0;
          s = Math.max(0, Math.min(1, s));
          /**
           * Redraws the four curves up to epoch tmax.
           */
          const render = (tmax) => {
            const n = Math.max(2, Math.round(tmax));
            paths.forEach((el, i) => {
              const fn = f[defs[i][0]];
              h.attr(el, { d: curvePath((t) => fn(t, s), 0, tmax, n, px, py, 0, VMAX) });
            });
            if (tmax >= T - 0.5) {
              const a = f.vaNo(T), b = f.trNo(T);
              h.attr(gapLine, { x1: px(T) + 8, x2: px(T) + 8, y1: py(a), y2: py(b) });
              h.attr(gapTxt, { x: px(T) - 112, y: (py(a) + py(b)) / 2 + 4 });
              gap.style.opacity = 1;
            } else gap.style.opacity = 0;
          };
          /**
           * Animates the curves being drawn from left to right.
           */
          const play = () => {
            const my = ++token;
            h.tween(3200, (p) => { if (my === token) render(Math.max(2, T * p)); }, h.ease.linear);
          };
          h.slider({ label: 'Probabilité $p$', min: 0.1, max: 0.8, step: 0.1, value: Math.max(0.1, Math.min(0.8, h.shared.dropP)), decimals: 1, onInput: (v) => { s = v / 0.8; token++; render(T); } });
          h.button('Rejouer l\'entraînement', play, 'primary');
          play();
        }
      }
    ]
  });
})();
