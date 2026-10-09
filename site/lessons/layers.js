/**
 * Lesson: the building blocks of a deep network (dense, batch norm, convolution, pooling).
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
   * Formats an integer with non-breaking spaces as thousands separators.
   */
  function big(n) {
    return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  }

  /**
   * Formats a number of float32 parameters as a memory size.
   */
  function memory(params) {
    const bytes = params * 4;
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1).replace('.', ',') + ' Mo';
    return (bytes / 1e3).toFixed(1).replace('.', ',') + ' Ko';
  }

  /**
   * Formats a number with an explicit sign handling suited for small cells.
   */
  function cellNum(h, v) {
    return Number.isInteger(v) ? h.fmt(v, 0) : h.fmt(v, 2);
  }

  const BLOCKS = [
    { name: 'Entrée', cap: '28×28', c: 'ink', hh: 150, desc: '<b>Entrée</b> : l\'image (ou le vecteur) que l\'on donne au réseau, par exemple $28\\times 28$ pixels.' },
    { name: 'Convolution', cap: 'filtres', c: 'accent', hh: 150, desc: '<b>Convolution</b> : des petits filtres appris glissent sur l\'image. Questions : quelle <em>taille</em> de filtres ? quel <em>pas</em> (stride) entre deux positions ?' },
    { name: 'BatchNorm', cap: 'N(0,1)', c: 'accent-2', hh: 150, desc: '<b>Batch normalisation</b> : normalise les activités pour qu\'elles suivent approximativement $\\mathcal{N}(0,1)$. Réduit le <em>vanishing gradient</em>.' },
    { name: 'ReLU', cap: 'max(0,x)', c: 'warn', hh: 150, desc: '<b>Non-linéarité</b> (tanh, sigmoïde, ReLU...) : sans elle, empiler des couches linéaires reviendrait à une seule couche linéaire.' },
    { name: 'Pooling', cap: '2×2', c: 'pos', hh: 100, desc: '<b>Pooling</b> : sous-échantillonne la carte de caractéristiques. Quelle stratégie (max, moyenne) ? Quel sous-échantillonnage ?' },
    { name: 'Dense', cap: 'Wx+b', c: 'neg', hh: 74, desc: '<b>Couche linéaire fully-connected</b> : $f(x) = Wx + b$. Tous les neurones d\'entrée sont connectés à tous les neurones de sortie.' },
    { name: 'Dropout', cap: 'p', c: 'bad', hh: 74, desc: '<b>Dropout</b> : ajoute du bruit à l\'apprentissage (régularisation) et réduit le sur-apprentissage.' },
    { name: 'Dense', cap: 'softmax', c: 'neg', hh: 44, desc: '<b>Dernière couche dense</b> + softmax : un score (probabilité) par classe.' }
  ];

  const DIGIT = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0]
  ];

  const FILTERS = {
    vert: { name: 'Bords verticaux', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], div: 1 },
    hor: { name: 'Bords horizontaux', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], div: 1 },
    blur: { name: 'Flou (moyenne)', k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], div: 9 }
  };

  const FEAT = [
    [1, 3, 2, 1, 0, 2],
    [4, 9, 1, 0, 3, 5],
    [2, 1, 0, 6, 1, 1],
    [0, 2, 7, 2, 2, 0],
    [1, 5, 1, 3, 8, 4],
    [3, 2, 0, 1, 2, 6]
  ];

  /**
   * Computes the valid (no padding) convolution of DIGIT with a filter at a given stride.
   */
  function convolve(filter, stride) {
    const f = FILTERS[filter];
    const n = Math.floor((7 - 3) / stride) + 1;
    const out = [];
    for (let oy = 0; oy < n; oy++) {
      const row = [];
      for (let ox = 0; ox < n; ox++) {
        let s = 0;
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += f.k[i][j] * DIGIT[oy * stride + i][ox * stride + j];
        row.push(s / f.div);
      }
      out.push(row);
    }
    return out;
  }

  Course.register({
    id: 'layers',
    order: 7,
    title: 'Les couches d\'un réseau profond',
    summary: 'Couche dense, batch normalisation, convolution et pooling : les briques que l\'on empile pour construire un réseau <em>profond</em>.',
    pdfPages: '27-29',
    steps: [
      {
        title: 'Panorama des couches',
        duration: 9000,
        text: '<p>Un réseau profond est un <strong>empilement de couches</strong>, chacune avec un rôle précis. On en combine plusieurs types :</p>' +
          '<ul><li>couche <strong>linéaire</strong> (fully-connected) $f(x)=Wx+b$ ;</li><li><strong>non-linéarité</strong> : tanh, sigmoïde, ReLU ;</li><li><strong>dropout</strong>, <strong>batch normalisation</strong> ;</li><li><strong>convolution</strong> et <strong>pooling</strong>.</li></ul>' +
          '<p>Cliquez sur un bloc pour lire son rôle. Les prochaines étapes détaillent chaque brique.</p>',
        note: 'Chaque couche transforme la représentation de la précédente ; la hauteur des blocs suggère la taille des données qui circulent.',
        draw(h) {
          const svg = h.svg(720, 420);
          const bw = 68, gap = 18, x0 = 25, cy = 175;
          const groups = [];
          const desc = h.label(svg, 20, 320, 680, 90, '', '');
          let selected = -1;
          /**
           * Highlights one block and shows its description.
           */
          const select = (i) => {
            selected = i;
            groups.forEach((g, k) => h.attr(g.rect, { 'stroke-width': k === i ? 3.5 : 1.5 }));
            desc.innerHTML = BLOCKS[i].desc;
            h.typeset(desc);
          };
          BLOCKS.forEach((b, i) => {
            const x = x0 + i * (bw + gap);
            const g = h.el('g', { style: 'opacity:0;transition:opacity .45s;cursor:pointer' }, svg);
            const rect = h.el('rect', {
              x, y: cy - b.hh / 2, width: bw, height: b.hh, rx: 8,
              fill: 'var(--' + b.c + ')', 'fill-opacity': 0.2, stroke: 'var(--' + b.c + ')', 'stroke-width': 1.5
            }, g);
            h.el('text', { x: x + bw / 2, y: cy + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--ink)', text: b.cap }, g);
            h.el('text', { x: x + bw / 2, y: cy + 112, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--ink)', text: b.name }, g);
            if (i < BLOCKS.length - 1) {
              h.el('line', { x1: x + bw + 2, y1: cy, x2: x + bw + gap - 2, y2: cy, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, g);
            }
            g.addEventListener('click', () => select(i));
            groups.push({ g, rect });
          });
          h.el('text', { x: 360, y: 28, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--muted)', text: 'Exemple d\'empilement pour classer des images' }, svg);
          (async () => {
            for (let i = 0; i < BLOCKS.length; i++) {
              await h.sleep(i === 0 ? 300 : 520);
              groups[i].g.style.opacity = 1;
              select(i);
            }
            await h.sleep(500);
            if (selected === BLOCKS.length - 1) select(5);
          })();
        },
        check: {
          q: 'Pourquoi insère-t-on une non-linéarité (ReLU, tanh...) entre deux couches linéaires ?',
          choices: [
            'Pour accélérer le calcul',
            'Parce que deux couches linéaires successives équivalent à une seule couche linéaire',
            'Pour réduire le nombre de paramètres'
          ],
          answer: 1,
          explain: '$W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2)$ : c\'est encore une fonction linéaire. La non-linéarité donne de la puissance de représentation.'
        }
      },
      {
        title: 'Couche dense',
        duration: 14000,
        text: '<p>Dans une couche <strong>dense</strong> (<em>fully connected</em>), <strong>chaque neurone d\'entrée est relié à chaque neurone de sortie</strong>.</p>' +
          '<p>L\'opération est linéaire : $$y = W\\,x + b$$ avec $W$ de taille $n_{out}\\times n_{in}$. Chaque sortie $y_i$ est le produit scalaire de la ligne $i$ de $W$ avec $x$, plus le biais $b_i$.</p>',
        note: 'Épaisseur d\'une arête = |poids| ; vert = poids positif, rouge = poids négatif.',
        draw(h) {
          const svg = h.svg(720, 420);
          const W = [[0.5, -1, 0.3, 2], [1, 0.2, -0.5, 0], [-0.7, 0.4, 1, 1.5]];
          const X = [1, -1, 2, 0.5];
          const B = [0.1, -0.2, 0.3];
          const Y = W.map((row, i) => row.reduce((s, w, j) => s + w * X[j], B[i]));
          const sub = ['₁', '₂', '₃', '₄'];
          const inY = [122, 184, 246, 308], outY = [137, 215, 293];
          const inX = 70, outX = 250;
          h.el('text', { x: 20, y: 28, 'font-size': 14, fill: 'var(--muted)', text: '4 entrées → 3 sorties : 12 poids + 3 biais' }, svg);
          const edges = [];
          const edgeLayer = h.el('g', {}, svg);
          for (let i = 0; i < 3; i++) {
            edges.push([]);
            for (let j = 0; j < 4; j++) {
              const w = W[i][j];
              const e = h.el('line', {
                x1: inX + 20, y1: inY[j], x2: outX - 20, y2: outY[i],
                stroke: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'stroke-width': 0.8 + 2.2 * Math.abs(w) / 2,
                'stroke-opacity': 0, 'stroke-linecap': 'round'
              }, edgeLayer);
              edges[i].push(e);
            }
          }
          inY.forEach((y, j) => {
            h.el('circle', { cx: inX, cy: y, r: 20, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            h.el('text', { x: inX, y: y + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--ink)', text: h.fmt(X[j], 1) }, svg);
            h.el('text', { x: 28, y: y + 5, 'text-anchor': 'middle', 'font-size': 15, fill: 'var(--ink)', text: 'x' + sub[j] }, svg);
          });
          const outTxt = [];
          outY.forEach((y, i) => {
            h.el('circle', { cx: outX, cy: y, r: 20, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
            outTxt.push(h.el('text', { x: outX, y: y + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--ink)', text: '?' }, svg));
            h.el('text', { x: outX + 36, y: y + 5, 'text-anchor': 'middle', 'font-size': 15, fill: 'var(--ink)', text: 'y' + sub[i] }, svg);
          });
          const cw = 40, ch = 34;
          const wx = 345, xx = 520, bx = 585, yx = 650;
          const topW = 164, topX = 147;
          const mk = (x, y, txt, parent) => {
            const g = h.el('g', { style: 'opacity:0;transition:opacity .5s' }, parent || svg);
            const rect = h.el('rect', { x, y, width: cw, height: ch, rx: 4, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1 }, g);
            const t = h.el('text', { x: x + cw / 2, y: y + 22, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--ink)', text: txt }, g);
            return { g, rect, t };
          };
          const mat = h.el('g', {}, svg);
          const wC = W.map((row, i) => row.map((w, j) => mk(wx + j * cw, topW + i * ch, h.fmt(w, 1), mat)));
          const xC = X.map((v, j) => mk(xx, topX + j * ch, h.fmt(v, 1), mat));
          const bC = B.map((v, i) => mk(bx, topW + i * ch, h.fmt(v, 1), mat));
          const yC = Y.map((v, i) => mk(yx, topW + i * ch, '', mat));
          const heads = [['W', wx + 2 * cw, 138], ['x', xx + cw / 2, 128], ['b', bx + cw / 2, 150], ['y', yx + cw / 2, 150]];
          const ops = [['×', 505, 220], ['+', 575, 220], ['=', 638, 220]];
          const decor = h.el('g', { style: 'opacity:0;transition:opacity .5s' }, svg);
          heads.forEach(([t, x, y]) => h.el('text', { x, y, 'text-anchor': 'middle', 'font-size': 16, fill: 'var(--accent)', 'font-weight': 700, text: t }, decor));
          ops.forEach(([t, x, y]) => h.el('text', { x, y, 'text-anchor': 'middle', 'font-size': 18, fill: 'var(--ink)', text: t }, decor));
          const calc = h.el('text', { x: 360, y: 372, 'text-anchor': 'middle', 'font-size': 12.5, class: 'mono', fill: 'var(--ink)', text: '' }, svg);
          const foot = h.el('text', { x: 360, y: 402, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: '' }, svg);
          let token = 0;
          /**
           * Resets edge styles to the neutral state.
           */
          const neutral = () => {
            edges.forEach((row) => row.forEach((e) => h.attr(e, { 'stroke-opacity': 0.75 })));
          };
          /**
           * Plays the whole animation: connections, then the matrix-vector product row by row.
           */
          const run = async () => {
            const my = ++token;
            edges.forEach((row) => row.forEach((e) => h.attr(e, { 'stroke-opacity': 0, stroke: e.getAttribute('stroke') })));
            [wC, [xC], [bC], [yC]].forEach((m) => m.forEach((r) => (Array.isArray(r) ? r : [r]).forEach((c) => (c.g.style.opacity = 0))));
            decor.style.opacity = 0;
            yC.forEach((c) => { c.t.textContent = ''; h.attr(c.rect, { fill: 'var(--surface-2)', stroke: 'var(--line)' }); });
            outTxt.forEach((t) => (t.textContent = '?'));
            calc.textContent = '';
            foot.textContent = 'Chaque entrée est reliée à chaque sortie';
            for (let j = 0; j < 4; j++) {
              for (let i = 0; i < 3; i++) {
                h.attr(edges[i][j], { 'stroke-opacity': 0.75 });
                await h.sleep(70);
                if (my !== token) return;
              }
            }
            await h.sleep(400);
            if (my !== token) return;
            decor.style.opacity = 1;
            wC.forEach((r) => r.forEach((c) => (c.g.style.opacity = 1)));
            xC.forEach((c) => (c.g.style.opacity = 1));
            bC.forEach((c) => (c.g.style.opacity = 1));
            yC.forEach((c) => (c.g.style.opacity = 1));
            foot.textContent = 'Le produit matrice-vecteur, ligne par ligne';
            await h.sleep(700);
            for (let i = 0; i < 3; i++) {
              if (my !== token) return;
              edges.forEach((row, k) => row.forEach((e) => h.attr(e, { 'stroke-opacity': k === i ? 1 : 0.12 })));
              wC[i].forEach((c) => h.attr(c.rect, { fill: 'var(--accent)', 'fill-opacity': 0.3, stroke: 'var(--accent)' }));
              xC.forEach((c) => h.attr(c.rect, { fill: 'var(--accent)', 'fill-opacity': 0.3, stroke: 'var(--accent)' }));
              h.attr(bC[i].rect, { fill: 'var(--accent)', 'fill-opacity': 0.3, stroke: 'var(--accent)' });
              const terms = W[i].map((w, j) => '(' + h.fmt(w, 1) + ')×(' + h.fmt(X[j], 1) + ')').join(' + ');
              calc.textContent = 'y' + sub[i] + ' = ' + terms + ' + (' + h.fmt(B[i], 1) + ') = ' + h.fmt(Y[i], 2);
              await h.sleep(1100);
              if (my !== token) return;
              yC[i].t.textContent = h.fmt(Y[i], 2);
              h.attr(yC[i].rect, { fill: 'var(--accent-2)', 'fill-opacity': 0.3, stroke: 'var(--accent-2)' });
              outTxt[i].textContent = h.fmt(Y[i], 1);
              wC[i].forEach((c) => h.attr(c.rect, { fill: 'var(--surface-2)', 'fill-opacity': 1, stroke: 'var(--line)' }));
              xC.forEach((c) => h.attr(c.rect, { fill: 'var(--surface-2)', 'fill-opacity': 1, stroke: 'var(--line)' }));
              h.attr(bC[i].rect, { fill: 'var(--surface-2)', 'fill-opacity': 1, stroke: 'var(--line)' });
              await h.sleep(400);
            }
            neutral();
            calc.textContent = '';
            foot.textContent = 'On applique ensuite souvent une non-linéarité à y';
          };
          h.button('Rejouer le calcul', run, 'primary');
          run();
        }
      },
      {
        title: 'Combien de paramètres ?',
        duration: 11000,
        text: '<p>Une couche dense possède <strong>un poids par connexion</strong> et un biais par sortie :</p>' +
          '$$n_{in}\\times n_{out} + n_{out}$$' +
          '<p>Comme le nombre de poids est le <em>produit</em> des tailles, il explose vite. Pour une image $28\\times 28$ aplatie ($784$ entrées), une seule couche de $512$ neurones coûte déjà plus de $400\\,000$ paramètres.</p>' +
          '<p>Déplacez les curseurs : l\'aire du rectangle représente la matrice $W$.</p>',
        note: 'Les couches denses sont très gourmandes en paramètres, ce qui motive les convolutions (poids partagés) pour les images.',
        check: {
          q: 'Combien de paramètres (poids + biais) pour une couche dense de 784 entrées vers 10 sorties ?',
          choices: ['7 840', '7 850', '794'],
          answer: 1,
          explain: '$784\\times 10 = 7840$ poids, auxquels s\'ajoutent $10$ biais : $7850$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const MAX = 4096, S = 300, ox = 70, oy = 40;
          h.el('rect', { x: ox, y: oy, width: S, height: S, fill: 'none', stroke: 'var(--line)', 'stroke-dasharray': '5 4' }, svg);
          h.el('text', { x: ox + S / 2, y: oy - 12, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)', text: 'référence : 4096 × 4096' }, svg);
          const area = h.el('rect', { x: ox, y: oy, width: 10, height: 10, fill: 'var(--accent)', 'fill-opacity': 0.35, stroke: 'var(--accent)', 'stroke-width': 2 }, svg);
          const bx = h.el('text', { x: ox + S / 2, y: oy + S + 28, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: '' }, svg);
          const by = h.el('text', { x: 36, y: oy + S / 2, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', transform: 'rotate(-90 36 ' + (oy + S / 2) + ')', text: '' }, svg);
          const panel = h.label(svg, 410, 50, 290, 300, '', '');
          let nIn = 784, nOut = 512;
          let sIn, sOut;
          /**
           * Refreshes the rectangle and the numeric panel.
           */
          const update = () => {
            const w = Math.max(3, S * nIn / MAX), hh = Math.max(3, S * nOut / MAX);
            h.attr(area, { width: w, height: hh });
            bx.textContent = 'nb_inputs = ' + big(nIn);
            by.textContent = 'nb_outputs = ' + big(nOut);
            const wts = nIn * nOut, total = wts + nOut;
            panel.innerHTML = '<p style="margin:0 0 8px"><b>Poids</b> : ' + big(nIn) + ' × ' + big(nOut) + ' = <b>' + big(wts) + '</b></p>' +
              '<p style="margin:0 0 8px"><b>Biais</b> : ' + big(nOut) + '</p>' +
              '<p style="margin:0 0 12px;font-size:1.2em"><b>Total : <span style="color:var(--accent)">' + big(total) + '</span></b></p>' +
              '<p style="margin:0;color:var(--muted)">Mémoire (float32) : ' + memory(total) + '</p>';
          };
          sIn = h.slider({ label: 'Entrées', min: 1, max: MAX, step: 1, value: nIn, format: (v) => big(v), onInput: (v) => { nIn = v; update(); } });
          sOut = h.slider({ label: 'Sorties', min: 1, max: MAX, step: 1, value: nOut, format: (v) => big(v), onInput: (v) => { nOut = v; update(); } });
          /**
           * Applies a preset size to both sliders.
           */
          const preset = (a, b) => { nIn = a; nOut = b; sIn.set(a); sOut.set(b); update(); };
          h.button('784 → 10', () => preset(784, 10));
          h.button('784 → 512', () => preset(784, 512));
          h.button('4096 → 4096', () => preset(4096, 4096));
          update();
        }
      },
      {
        title: 'Classifieur ou représentation',
        duration: 12000,
        text: '<p>Une couche dense seule est un <strong>classifieur linéaire</strong> : sa frontière de décision est un hyperplan. Elle échoue sur un problème de type XOR.</p>' +
          '<p>Mais utilisée <strong>comme couche de représentation</strong>, suivie d\'une non-linéarité, elle transforme les données pour qu\'une dernière couche puisse les séparer :</p>' +
          '$$y = W_2\\,\\mathrm{ReLU}(W_1x + b_1) + b_2$$' +
          '<p>Ici $h_1=\\mathrm{ReLU}(x_1+x_2)$ et $h_2=\\mathrm{ReLU}(-x_1-x_2)$ ; la sortie $h_1+h_2-1$ revient à tester $|x_1+x_2|>1$.</p>',
        note: 'Couche dense + non-linéarité : on apprend une représentation dans laquelle le problème devient (presque) linéaire.',
        draw(h) {
          const svg = h.svg(720, 420);
          const r = makeRng(h, 11);
          const px = h.scale(-2, 2, 30, 390), py = h.scale(-2, 2, 390, 30);
          const pts = [];
          [[1, 1], [-1, -1], [1, -1], [-1, 1]].forEach(([cx, cy]) => {
            for (let i = 0; i < 12; i++) {
              pts.push({ x: cx + (r() - 0.5) * 0.9, y: cy + (r() - 0.5) * 0.9, lab: cx * cy > 0 ? 1 : -1 });
            }
          });
          const lin = (x, y) => (x + 0.3 * y > 0 ? 1 : -1);
          const deep = (x, y) => (Math.max(0, x + y) + Math.max(0, -x - y) - 1 > 0 ? 1 : -1);
          const bg = h.el('g', {}, svg);
          const n = 24, cs = 360 / n;
          const cells = [];
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              cells.push({ x: -2 + (i + 0.5) * 4 / n, y: 2 - (j + 0.5) * 4 / n, el: h.el('rect', { x: 30 + i * cs, y: 30 + j * cs, width: cs + 0.4, height: cs + 0.4, 'fill-opacity': 0.2 }, bg) });
            }
          }
          h.el('rect', { x: 30, y: 30, width: 360, height: 360, fill: 'none', stroke: 'var(--line)' }, svg);
          pts.forEach((p) => h.el('circle', { cx: px(p.x), cy: py(p.y), r: 5, fill: p.lab > 0 ? 'var(--pos)' : 'var(--neg)', stroke: 'var(--ink)', 'stroke-width': 1 }, svg));
          const card = h.label(svg, 420, 60, 280, 280, '', '');
          const modeNames = {
            lin: '<b>Couche dense seule</b><br>$y = Wx + b$<br>Frontière : une droite.',
            deep: '<b>Dense + ReLU + dense</b><br>$y = W_2\\,\\mathrm{ReLU}(W_1x+b_1)+b_2$<br>Frontière : deux droites parallèles.'
          };
          /**
           * Colours the background and updates the accuracy for the selected model.
           */
          const show = (mode) => {
            const f = mode === 'lin' ? lin : deep;
            cells.forEach((c) => h.attr(c.el, { fill: f(c.x, c.y) > 0 ? 'var(--pos)' : 'var(--neg)' }));
            const ok = pts.filter((p) => f(p.x, p.y) === p.lab).length;
            card.innerHTML = '<p style="margin:0 0 12px">' + modeNames[mode] + '</p><p style="margin:0">Exemples bien classés : <b style="color:' + (ok === pts.length ? 'var(--ok)' : 'var(--bad)') + '">' + ok + ' / ' + pts.length + '</b> (' + Math.round(100 * ok / pts.length) + '&nbsp;%)</p>';
            h.typeset(card);
          };
          h.toggle([['lin', 'Dense seule'], ['deep', 'Dense + ReLU + dense']], 'lin', show);
          show('lin');
        }
      },
      {
        title: 'Batch normalisation',
        duration: 14000,
        text: '<p>Au fil des couches, la distribution des activités dérive (décalage, échelle). Les non-linéarités saturent, les gradients s\'évanouissent.</p>' +
          '<p>La <strong>batch normalisation</strong> normalise les activités du minibatch : on soustrait la moyenne puis on divise par l\'écart-type,</p>' +
          '$$\\hat{x} = \\frac{x-\\mu_B}{\\sqrt{\\sigma_B^2+\\varepsilon}}\\qquad y=\\gamma\\hat{x}+\\beta$$' +
          '<p>où $\\gamma$ et $\\beta$ sont <em>appris</em> : le réseau peut choisir l\'échelle et le décalage les plus utiles.</p>',
        note: 'Activités ~ N(0,1) : on évite la saturation des non-linéarités, ce qui <strong>réduit le vanishing gradient</strong>.',
        check: {
          q: 'Après normalisation (avant $\\gamma$ et $\\beta$), quelles sont la moyenne et la variance des activités du minibatch ?',
          choices: ['Moyenne 0, variance 1', 'Moyenne 1, variance 0', 'Moyenne 0, variance 0'],
          answer: 0,
          explain: 'On centre (moyenne 0) puis on divise par l\'écart-type (variance 1) : $\\hat{x}\\sim\\mathcal{N}(0,1)$.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const r = makeRng(h, 5);
          const N = 300;
          const xs = [];
          for (let i = 0; i < N; i++) {
            const u = Math.max(1e-6, r()), v = r();
            xs.push(3 + 2 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
          }
          const mu = xs.reduce((a, b) => a + b, 0) / N;
          const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / N);
          const X0 = -8, X1 = 12, BW = 0.5, NB = 40;
          const px = h.scale(X0, X1, 60, 680);
          const base = 330, per = 2.7;
          h.el('line', { x1: 55, y1: base, x2: 690, y2: base, stroke: 'var(--ink)', 'stroke-width': 1.5 }, svg);
          for (let v = X0; v <= X1; v += 2) {
            h.el('line', { x1: px(v), y1: base, x2: px(v), y2: base + 5, stroke: 'var(--ink)' }, svg);
            h.el('text', { x: px(v), y: base + 20, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--muted)', text: String(v).replace('-', '−') }, svg);
          }
          const band = h.el('rect', { x: 0, y: 70, width: 0, height: base - 70, fill: 'var(--accent)', 'fill-opacity': 0.12 }, svg);
          const bars = [];
          for (let i = 0; i < NB; i++) {
            bars.push(h.el('rect', { x: px(X0 + i * BW) + 1, y: base, width: px(X0 + BW) - px(X0) - 2, height: 0, fill: 'var(--accent)', 'fill-opacity': 0.65, stroke: 'var(--accent)' }, svg));
          }
          const mline = h.el('line', { x1: 0, y1: 70, x2: 0, y2: base, stroke: 'var(--accent-2)', 'stroke-width': 2.5, 'stroke-dasharray': '6 4' }, svg);
          h.el('text', { x: 60, y: 20, 'font-size': 13, fill: 'var(--muted)', text: 'Activités d\'un neurone sur un minibatch de ' + N + ' exemples' }, svg);
          const phase = h.el('text', { x: 680, y: 20, 'text-anchor': 'end', 'font-size': 14, fill: 'var(--accent-2)', 'font-weight': 700, text: '' }, svg);
          h.el('text', { x: 680, y: 62, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--accent-2)', text: 'moyenne (trait) · ± 1 écart-type (bande)' }, svg);
          const ro = h.readout('');
          let t = 0, gam = 1, bet = 0, token = 0;
          /**
           * Redraws the histogram for the current progress t and the parameters gamma and beta.
           */
          const render = () => {
            const a = Math.min(1, t * 2), b = Math.max(0, t * 2 - 1);
            const s = h.lerp(1, 1 / sd, b);
            const g = h.lerp(1, gam, b), be = bet * b;
            const vals = xs.map((x) => g * (x - mu * a) * s + be);
            const counts = new Array(NB).fill(0);
            vals.forEach((v) => {
              const k = Math.floor((v - X0) / BW);
              if (k >= 0 && k < NB) counts[k]++;
            });
            counts.forEach((c, i) => {
              const hh = Math.min(base - 70, c * per);
              h.attr(bars[i], { y: base - hh, height: hh });
            });
            const m = vals.reduce((p, q) => p + q, 0) / N;
            const v = vals.reduce((p, q) => p + (q - m) * (q - m), 0) / N;
            const w = Math.sqrt(v);
            h.attr(mline, { x1: px(m), x2: px(m) });
            h.attr(band, { x: px(m - w), width: Math.max(0, px(m + w) - px(m - w)) });
            phase.textContent = t <= 0.001 ? 'Avant normalisation' : t < 0.5 ? 'Étape 1 : on centre' : t < 0.999 ? 'Étape 2 : on réduit l\'échelle' : (gam === 1 && bet === 0 ? 'Après normalisation' : 'Après γ et β');
            ro('Moyenne $\\mu = ' + h.fmt(m, 2) + '$ · variance $\\sigma^2 = ' + h.fmt(v, 2) + '$');
          };
          /**
           * Animates the progress to a target value (1 = normalised, 0 = raw).
           */
          const go = (to) => {
            const my = ++token, from = t;
            h.tween(2600, (p) => { if (my === token) { t = from + (to - from) * p; render(); } });
          };
          h.button('Normaliser', () => go(1), 'primary');
          h.button('Réinitialiser', () => go(0));
          h.slider({ label: '$\\gamma$', min: 0.7, max: 2, step: 0.05, value: 1, onInput: (v) => { gam = v; render(); } });
          h.slider({ label: '$\\beta$', min: -3, max: 3, step: 0.1, value: 0, decimals: 1, onInput: (v) => { bet = v; render(); } });
          render();
          h.after(900, () => go(1));
        }
      },
      {
        title: 'Convolution',
        duration: 20000,
        text: '<p>Au lieu de relier chaque pixel à chaque neurone, la <strong>convolution</strong> fait glisser un petit <strong>filtre</strong> (ici $3\\times 3$) sur l\'image. À chaque position : somme des produits pixel $\\times$ poids, qui donne une case de la <strong>carte de caractéristiques</strong>.</p>' +
          '<p>Deux choix de conception :</p>' +
          '<ul><li>la <strong>taille des filtres</strong> ;</li><li>le <strong>pas</strong> (stride) entre deux positions du filtre.</li></ul>' +
          '<p>Sans bordure, la sortie a pour côté $\\lfloor (n-k)/s\\rfloor + 1$. Changez de filtre ou de stride.</p>',
        note: 'Le même filtre (mêmes poids) est utilisé partout dans l\'image : peu de paramètres, et détection de motifs où qu\'ils soient.',
        check: {
          q: 'Image $7\\times 7$, filtre $3\\times 3$, stride $2$, sans bordure. Quelle est la taille de la sortie ?',
          choices: ['$5\\times 5$', '$3\\times 3$', '$2\\times 2$'],
          answer: 1,
          explain: '$\\lfloor (7-3)/2\\rfloor + 1 = 3$ : le filtre peut se placer en colonnes 0, 2 et 4.'
        },
        draw(h) {
          const svg = h.svg(720, 420);
          const CS = 36, IX = 28, IY = 62;
          h.el('text', { x: IX + 126, y: 40, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Image 7×7' }, svg);
          for (let i = 0; i < 7; i++) {
            for (let j = 0; j < 7; j++) {
              const v = DIGIT[i][j];
              h.el('rect', { x: IX + j * CS, y: IY + i * CS, width: CS, height: CS, fill: 'var(--ink)', 'fill-opacity': v ? 0.85 : 0.04, stroke: 'var(--line)' }, svg);
              h.el('text', { x: IX + j * CS + CS / 2, y: IY + i * CS + 23, 'text-anchor': 'middle', 'font-size': 11, class: 'mono', fill: v ? 'var(--surface)' : 'var(--muted)', text: String(v) }, svg);
            }
          }
          const win = h.el('rect', { x: IX, y: IY, width: 3 * CS, height: 3 * CS, fill: 'var(--accent)', 'fill-opacity': 0.12, stroke: 'var(--accent)', 'stroke-width': 3.5 }, svg);
          const FX = 322, FY = 150, FC = 30;
          h.el('text', { x: FX + 45, y: 40, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Filtre 3×3' }, svg);
          const fg = h.el('g', {}, svg);
          const sum = h.el('text', { x: FX + 45, y: FY + 134, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--ink)', text: '' }, svg);
          h.el('line', { x1: FX + 100, y1: FY + 45, x2: 424, y2: FY + 45, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
          const OX = 440, OY = 62, OS = 252;
          const outTitle = h.el('text', { x: OX + OS / 2, y: 40, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: '' }, svg);
          const og = h.el('g', {}, svg);
          const foot = h.el('text', { x: 360, y: 396, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: '' }, svg);
          let key = 'vert', stride = 1, token = 0;
          /**
           * Draws the 3x3 filter weights.
           */
          const drawFilter = () => {
            clear(fg);
            const f = FILTERS[key];
            for (let i = 0; i < 3; i++) {
              for (let j = 0; j < 3; j++) {
                const w = f.k[i][j] / f.div;
                h.el('rect', { x: FX + j * FC, y: FY + i * FC, width: FC, height: FC, fill: w >= 0 ? 'var(--pos)' : 'var(--neg)', 'fill-opacity': w === 0 ? 0.04 : 0.15 + 0.4 * Math.abs(w), stroke: 'var(--line)' }, fg);
                h.el('text', { x: FX + j * FC + FC / 2, y: FY + i * FC + 20, 'text-anchor': 'middle', 'font-size': 11, class: 'mono', fill: 'var(--ink)', text: cellNum(h, Math.round(w * 100) / 100) }, fg);
              }
            }
          };
          /**
           * Restarts the sliding-window animation for the current filter and stride.
           */
          const run = async () => {
            const my = ++token;
            const out = convolve(key, stride);
            const n = out.length;
            const maxAbs = Math.max(0.001, ...out.map((row) => Math.max(...row.map(Math.abs))));
            const cs = Math.min(60, OS / n);
            const ox0 = OX + (OS - cs * n) / 2, oy0 = OY + (OS - cs * n) / 2;
            clear(og);
            drawFilter();
            outTitle.textContent = 'Carte de caractéristiques ' + n + '×' + n;
            sum.textContent = '';
            foot.textContent = 'Taille de sortie : ⌊(7 − 3) / ' + stride + '⌋ + 1 = ' + n;
            const cells = [];
            for (let i = 0; i < n; i++) {
              cells.push([]);
              for (let j = 0; j < n; j++) {
                const g = h.el('g', {}, og);
                const rc = h.el('rect', { x: ox0 + j * cs, y: oy0 + i * cs, width: cs, height: cs, fill: 'var(--surface-2)', stroke: 'var(--line)' }, g);
                const tx = h.el('text', { x: ox0 + j * cs + cs / 2, y: oy0 + i * cs + cs / 2 + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'mono', fill: 'var(--ink)', text: '' }, g);
                cells[i].push({ rc, tx });
              }
            }
            for (let i = 0; i < n; i++) {
              for (let j = 0; j < n; j++) {
                h.attr(win, { x: IX + j * stride * CS, y: IY + i * stride * CS });
                const v = out[i][j];
                h.attr(cells[i][j].rc, { stroke: 'var(--accent)', 'stroke-width': 3 });
                sum.textContent = 'Σ pixel × poids = ' + cellNum(h, Math.round(v * 100) / 100);
                await h.sleep(stride === 1 ? 420 : 650);
                if (my !== token) return;
                cells[i][j].tx.textContent = cellNum(h, Math.round(v * 100) / 100);
                h.attr(cells[i][j].rc, {
                  fill: v >= 0 ? 'var(--pos)' : 'var(--neg)',
                  'fill-opacity': v === 0 ? 0.05 : 0.15 + 0.7 * Math.abs(v) / maxAbs,
                  stroke: 'var(--line)', 'stroke-width': 1
                });
              }
            }
            sum.textContent = '';
          };
          h.toggle([['vert', 'Bords verticaux'], ['hor', 'Bords horizontaux'], ['blur', 'Flou']], key, (v) => { key = v; run(); });
          h.slider({ label: 'Stride', min: 1, max: 3, step: 1, value: 1, decimals: 0, onInput: (v) => { stride = v; run(); } });
          h.button('Relancer', run, 'primary');
          run();
        }
      },
      {
        title: 'Pooling',
        duration: 15000,
        text: '<p>Le <strong>pooling</strong> résume chaque petit bloc de la carte de caractéristiques par <em>une seule valeur</em> : c\'est un <strong>sous-échantillonnage</strong>.</p>' +
          '<ul><li><strong>Max pooling</strong> : on garde la valeur la plus forte (présence du motif) ;</li><li><strong>Average pooling</strong> : on garde la moyenne.</li></ul>' +
          '<p>Avec un bloc $2\\times 2$ et un pas de $2$, une carte $6\\times 6$ devient $3\\times 3$ : quatre fois moins de valeurs, et une petite invariance aux décalages.</p>',
        note: 'Le pooling n\'a aucun paramètre à apprendre : seule la stratégie (max, moyenne) et la taille du sous-échantillonnage sont à choisir.',
        draw(h) {
          const svg = h.svg(720, 420);
          const CS = 44, FX = 50, FY = 80, OS = 70, OX = 430, OY = 124;
          h.el('text', { x: FX + 3 * CS, y: 52, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Carte 6×6' }, svg);
          const maxV = 9;
          for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
              h.el('rect', { x: FX + j * CS, y: FY + i * CS, width: CS, height: CS, fill: 'var(--accent)', 'fill-opacity': 0.06 + 0.5 * FEAT[i][j] / maxV, stroke: 'var(--line)' }, svg);
              h.el('text', { x: FX + j * CS + CS / 2, y: FY + i * CS + 27, 'text-anchor': 'middle', 'font-size': 14, class: 'mono', fill: 'var(--ink)', text: String(FEAT[i][j]) }, svg);
            }
          }
          const win = h.el('rect', { x: FX, y: FY, width: 2 * CS, height: 2 * CS, fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 4 }, svg);
          const pick = h.el('rect', { x: FX, y: FY, width: CS, height: CS, fill: 'none', stroke: 'var(--ok)', 'stroke-width': 3, 'stroke-opacity': 0 }, svg);
          h.el('text', { x: OX + 1.5 * OS, y: 52, 'text-anchor': 'middle', 'font-size': 14, fill: 'var(--ink)', text: 'Sortie 3×3' }, svg);
          h.el('line', { x1: FX + 6 * CS + 14, y1: 212, x2: OX - 14, y2: 212, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, svg);
          h.el('text', { x: (FX + 6 * CS + OX) / 2, y: 200, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)', text: 'pool 2×2, pas 2' }, svg);
          const og = h.el('g', {}, svg);
          const foot = h.el('text', { x: 360, y: 396, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)', text: '36 valeurs → 9 valeurs (÷ 4)' }, svg);
          const calc = h.el('text', { x: OX + 1.5 * OS, y: 366, 'text-anchor': 'middle', 'font-size': 13, class: 'mono', fill: 'var(--ink)', text: '' }, svg);
          let mode = 'max', token = 0;
          /**
           * Restarts the pooling animation for the current strategy.
           */
          const run = async () => {
            const my = ++token;
            clear(og);
            h.attr(pick, { 'stroke-opacity': 0 });
            calc.textContent = '';
            const cells = [];
            for (let i = 0; i < 3; i++) {
              cells.push([]);
              for (let j = 0; j < 3; j++) {
                const rc = h.el('rect', { x: OX + j * OS, y: OY + i * OS, width: OS, height: OS, fill: 'var(--surface-2)', stroke: 'var(--line)' }, og);
                const tx = h.el('text', { x: OX + j * OS + OS / 2, y: OY + i * OS + OS / 2 + 6, 'text-anchor': 'middle', 'font-size': 18, class: 'mono', fill: 'var(--ink)', text: '' }, og);
                cells[i].push({ rc, tx });
              }
            }
            for (let i = 0; i < 3; i++) {
              for (let j = 0; j < 3; j++) {
                h.attr(win, { x: FX + j * 2 * CS, y: FY + i * 2 * CS });
                h.attr(cells[i][j].rc, { stroke: 'var(--accent-2)', 'stroke-width': 3 });
                const vals = [FEAT[2 * i][2 * j], FEAT[2 * i][2 * j + 1], FEAT[2 * i + 1][2 * j], FEAT[2 * i + 1][2 * j + 1]];
                let best = 0;
                vals.forEach((v, k) => { if (v > vals[best]) best = k; });
                const res = mode === 'max' ? vals[best] : vals.reduce((a, b) => a + b, 0) / 4;
                calc.textContent = mode === 'max' ? 'max(' + vals.join(', ') + ') = ' + res : 'moyenne(' + vals.join(', ') + ') = ' + h.fmt(res, 2);
                h.attr(pick, { 'stroke-opacity': 0 });
                await h.sleep(500);
                if (my !== token) return;
                if (mode === 'max') {
                  h.attr(pick, { x: FX + (2 * j + (best % 2)) * CS, y: FY + (2 * i + Math.floor(best / 2)) * CS, 'stroke-opacity': 1 });
                  await h.sleep(450);
                  if (my !== token) return;
                }
                cells[i][j].tx.textContent = mode === 'max' ? String(res) : h.fmt(res, 2);
                h.attr(cells[i][j].rc, { fill: 'var(--accent-2)', 'fill-opacity': 0.12 + 0.45 * res / maxV, stroke: 'var(--line)', 'stroke-width': 1 });
                await h.sleep(250);
                if (my !== token) return;
              }
            }
            h.attr(pick, { 'stroke-opacity': 0 });
            calc.textContent = '';
          };
          h.toggle([['max', 'Max pooling'], ['avg', 'Average pooling']], mode, (v) => { mode = v; run(); });
          h.button('Relancer', run, 'primary');
          run();
        },
        check: {
          q: 'Quel est le principal effet d\'un pooling $2\\times 2$ de pas $2$ ?',
          choices: [
            'Il divise par 4 le nombre de valeurs de la carte (sous-échantillonnage)',
            'Il ajoute des paramètres à apprendre',
            'Il augmente la résolution de la carte'
          ],
          answer: 0,
          explain: 'Chaque bloc $2\\times2$ est résumé par une valeur : la carte perd la moitié de sa hauteur et de sa largeur, sans aucun paramètre appris.'
        }
      }
    ]
  });
})();
