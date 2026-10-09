/**
 * Lesson: unsupervised pre-training with auto-encoders, greedy layer-wise training, finetuning and transfer learning.
 */
(function () {
  /**
   * Adds a centred SVG text element.
   */
  function text(h, parent, x, y, s, attrs) {
    return h.el('text', Object.assign({ x, y, 'text-anchor': 'middle', text: s }, attrs || {}), parent);
  }

  /**
   * Fades an SVG element from transparent to opaque.
   */
  function appear(h, el, ms) {
    return h.tween(ms || 350, (p) => el.setAttribute('opacity', p));
  }

  /**
   * Adds an arrow line, hidden until animated.
   */
  function arrow(h, parent, x1, y1, x2, y2) {
    return h.el('line', { x1, y1, x2, y2, stroke: 'var(--muted)', 'stroke-width': 1.8, 'marker-end': 'url(#arrow)', opacity: 0 }, parent);
  }

  /**
   * Draws a small padlock centred horizontally at x, with its top at y, hidden until shown.
   */
  function lock(h, parent, x, y) {
    const g = h.el('g', { transform: `translate(${x} ${y})`, opacity: 0 }, parent);
    h.el('rect', { x: -7, y: 0, width: 14, height: 11, rx: 2, fill: 'var(--ink)' }, g);
    h.el('path', { d: 'M -4 0 V -4 A 4 4 0 0 1 4 -4 V 0', fill: 'none', stroke: 'var(--ink)', 'stroke-width': 2 }, g);
    return g;
  }

  /**
   * Builds a named layer block. Returns the group, rectangle and label.
   */
  function block(h, parent, x, y, w, hh, label) {
    const g = h.el('g', {}, parent);
    const r = h.el('rect', { x, y, width: w, height: hh, rx: 10, fill: 'var(--surface)', stroke: 'var(--line)', 'stroke-width': 2 }, g);
    const t = text(h, g, x + w / 2, y + hh / 2 + 5, label, { 'font-weight': 600 });
    return { g, r, t };
  }

  /**
   * Paints a layer block as train (accent), frozen (muted) or todo (dashed).
   */
  function paint(b, state) {
    const map = {
      train: ['var(--accent)', 'var(--accent)', 0.3, 'none'],
      frozen: ['var(--surface-2)', 'var(--muted)', 1, 'none'],
      todo: ['var(--surface)', 'var(--line)', 1, '6 4']
    };
    const [fill, stroke, op, dash] = map[state];
    b.r.setAttribute('fill', fill);
    b.r.setAttribute('stroke', stroke);
    b.r.setAttribute('fill-opacity', op);
    b.r.setAttribute('stroke-dasharray', dash);
    b.t.setAttribute('style', state === 'frozen' ? 'fill: var(--muted)' : 'fill: var(--ink)');
  }

  /**
   * Draws unlabelled and labelled data side by side, linked by the pre-training idea.
   */
  function drawSupervised(h) {
    const s = h.svg(720, 420);
    const rng = h.rng(42);
    const samples = [];
    for (let i = 0; i < 40; i++) {
      const cls = i % 2;
      const cu = cls ? 0.4 : -0.4;
      const cv = cls ? 0.3 : -0.3;
      samples.push({ cls, u: cu + (rng() - 0.5) * 0.7, v: cv + (rng() - 0.5) * 0.7 });
    }
    h.el('rect', { x: 40, y: 80, width: 300, height: 260, rx: 14, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.5 }, s);
    h.el('rect', { x: 380, y: 80, width: 300, height: 260, rx: 14, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.5 }, s);
    text(h, s, 190, 68, 'Supervisé : données étiquetées', { 'font-weight': 700 });
    text(h, s, 530, 68, 'Non supervisé : données brutes', { 'font-weight': 700 });

    const left = samples.map((p) => h.el('circle', {
      cx: 190 + p.u * 120, cy: 205 + p.v * 100, r: 6, opacity: 0,
      fill: p.cls ? 'var(--neg)' : 'var(--pos)'
    }, s));
    const right = samples.map((p) => h.el('circle', {
      cx: 530 + p.u * 120, cy: 205 + p.v * 100, r: 6, opacity: 0, fill: 'var(--muted)'
    }, s));
    h.el('circle', { cx: 120, cy: 360, r: 6, fill: 'var(--pos)' }, s);
    text(h, s, 150, 364, 'classe A', { 'text-anchor': 'start' });
    h.el('circle', { cx: 240, cy: 360, r: 6, fill: 'var(--neg)' }, s);
    text(h, s, 270, 364, 'classe B', { 'text-anchor': 'start' });
    text(h, s, 530, 364, 'aucune étiquette, seulement les entrées', { class: 'muted' });
    text(h, s, 360, 394, "Pré-apprentissage : les couches apprennent d'abord sur des données non étiquetées", { class: 'muted' });

    const run = async () => {
      for (let i = 0; i < samples.length; i++) {
        left[i].setAttribute('opacity', 1);
        right[i].setAttribute('opacity', 0.85);
        await h.sleep(30);
      }
    };
    run();
  }

  /**
   * Draws an auto-encoder whose reconstruction error decreases over animated epochs.
   */
  function drawAutoencoder(h) {
    const s = h.svg(720, 420);
    const rng = h.rng(5);
    const inX = 110, codeX = 360, outX = 610;
    const inY = (i) => 70 + i * 40;
    const codeY = [170, 210, 250];
    const vals = Array.from({ length: 8 }, () => 0.15 + 0.85 * rng());
    const start = Array.from({ length: 8 }, () => rng());

    const net = h.el('g', { opacity: 0 }, s);
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 3; j++) {
        h.el('line', { x1: inX, y1: inY(i), x2: codeX, y2: codeY[j], stroke: 'var(--muted)', 'stroke-width': 1, opacity: 0.35 }, net);
        h.el('line', { x1: codeX, y1: codeY[j], x2: outX, y2: inY(i), stroke: 'var(--muted)', 'stroke-width': 1, opacity: 0.35 }, net);
      }
    }
    const inNodes = vals.map((v, i) => h.el('circle', { cx: inX, cy: inY(i), r: 14, fill: 'var(--accent)', 'fill-opacity': v, stroke: 'var(--ink)', 'stroke-width': 1.5 }, net));
    codeY.forEach((y) => h.el('circle', { cx: codeX, cy: y, r: 16, fill: 'var(--accent-2)', stroke: 'var(--ink)', 'stroke-width': 1.5 }, net));
    const outNodes = start.map((v, i) => h.el('circle', { cx: outX, cy: inY(i), r: 14, fill: 'var(--accent)', 'fill-opacity': v, stroke: 'var(--ink)', 'stroke-width': 1.5 }, net));
    text(h, net, inX, 50, 'entrée x');
    text(h, net, codeX, 130, 'code (goulot)');
    text(h, net, outX, 50, 'reconstruction x̂');
    text(h, net, (inX + codeX) / 2, 385, 'encodeur', { class: 'muted' });
    text(h, net, (codeX + outX) / 2, 385, 'décodeur', { class: 'muted' });
    const errTxt = text(h, s, 360, 22, '', { 'font-weight': 700, class: 'mono' });

    const mse = () => {
      let e = 0;
      vals.forEach((v, i) => { e += (v - outNodes[i].rv) ** 2; });
      return e / vals.length;
    };

    const show = (p, epoch) => {
      outNodes.forEach((n, i) => {
        const v = start[i] + (vals[i] - start[i]) * p;
        n.rv = v;
        n.setAttribute('fill-opacity', v);
      });
      errTxt.textContent = `époque ${epoch} · erreur de reconstruction ${h.fmt(mse(), 3)}`;
    };

    const run = async () => {
      outNodes.forEach((n, i) => { n.rv = start[i]; });
      show(0, 0);
      await appear(h, net, 600);
      for (let e = 1; e <= 12; e++) {
        await h.sleep(400);
        show(1 - Math.pow(0.7, e), e);
      }
    };
    run();
  }

  /**
   * Draws greedy layer-wise pre-training: each layer is trained then frozen before the next one.
   */
  function drawGreedy(h) {
    const s = h.svg(720, 420);
    const cols = [30, 260, 490];
    const B = cols.map((cx, c) => {
      const layers = [
        block(h, s, cx + 35, 250, 130, 50, 'couche 1'),
        block(h, s, cx + 35, 180, 130, 50, 'couche 2'),
        block(h, s, cx + 35, 110, 130, 50, 'couche 3')
      ];
      if (c === 2) layers.push(block(h, s, cx + 35, 40, 130, 50, 'sortie'));
      layers.forEach((b) => paint(b, 'todo'));
      return layers;
    });
    const captions = [
      ['1. Auto-encodeur sur la couche 1', '(données sans étiquettes)'],
      ['2. Couche 1 gelée, auto-encodeur', 'sur la couche 2'],
      ['3. Empilement puis finetuning', '(avec étiquettes)']
    ];
    captions.forEach(([a, b], c) => {
      text(h, s, cols[c] + 100, 330, a, { class: 'muted', 'font-size': 12 });
      text(h, s, cols[c] + 100, 348, b, { class: 'muted', 'font-size': 12 });
    });
    text(h, s, 360, 392, 'Pré-entraînement glouton : une couche à la fois, puis empilement', { class: 'muted' });

    const run = async () => {
      paint(B[0][0], 'train');
      await h.sleep(1400);
      paint(B[0][0], 'frozen');
      paint(B[1][0], 'frozen');
      paint(B[1][1], 'train');
      await h.sleep(1400);
      paint(B[1][1], 'frozen');
      paint(B[2][0], 'frozen');
      paint(B[2][1], 'frozen');
      paint(B[2][2], 'train');
      paint(B[2][3], 'train');
      await h.sleep(1400);
      paint(B[2][0], 'train');
      paint(B[2][1], 'train');
      paint(B[2][2], 'train');
    };
    run();
  }

  /**
   * Draws finetuning of a pre-trained network with frozen layers (padlocks) and a new trainable head.
   */
  function drawFinetune(h) {
    const s = h.svg(720, 420);
    const xs = [40, 200, 360, 520];
    const names = ['couche 1', 'couche 2', 'couche 3', 'nouvelle couche'];
    const subs = ['bords, textures', 'motifs', `parties d'objets`, 'tâche cible'];
    const blocks = xs.map((x, i) => block(h, s, x, 130, 120, 110, names[i]));
    const locks = xs.map((x) => lock(h, s, x + 60, 142));
    xs.forEach((x, i) => {
      if (i < 3) arrow(h, s, x + 124, 185, xs[i + 1] - 4, 185).setAttribute('opacity', 1);
      text(h, s, x + 60, 262, subs[i], { class: 'muted' });
    });
    const caption = h.label(s, 40, 290, 640, 110, '', '');
    const setLoss = h.readout('');

    let mode = 'geler';
    /**
     * Applies the current mode (freeze or finetune) to every block and to the padlocks.
     */
    const apply = () => {
      const geler = mode === 'geler';
      blocks.forEach((b, i) => {
        paint(b, i === 3 || !geler ? 'train' : 'frozen');
        locks[i].setAttribute('opacity', geler && i < 3 ? 1 : 0);
      });
      caption.innerHTML = geler
        ? `<b>Geler</b> : les couches 1 à 3 gardent leurs poids (cadenas). Seule la nouvelle couche est entraînée, ce qui convient à un <b>petit</b> jeu de données.`
        : `<b>Affiner</b> : toutes les couches sont ajustées, avec un petit taux d'apprentissage pour ne pas effacer ce qui a été appris. Il faut davantage de données.`;
      h.typeset(caption);
    };

    h.toggle([['geler', 'Geler les couches'], ['affiner', 'Affiner tout']], 'geler', (v) => {
      mode = v;
      apply();
    });

    h.loop((t) => {
      const pulse = 0.5 + 0.5 * Math.sin(t / 250);
      const trainable = mode === 'geler' ? [blocks[3]] : blocks;
      trainable.forEach((b) => b.r.setAttribute('fill-opacity', 0.15 + 0.3 * pulse));
      const ep = Math.floor(t / 900) % 12;
      const floor = mode === 'geler' ? 0.45 : 0.08;
      const loss = floor + (1 - floor) * Math.pow(0.72, ep);
      setLoss(`époque ${ep} · perte (illustrative) ${h.fmt(loss, 2)}`);
    });

    apply();
    blocks.forEach((b, i) => { if (i < 3) appear(h, b.g, 300); });
  }

  /**
   * Draws a network where a slider chooses how many first layers are frozen.
   */
  function drawSummary(h) {
    const s = h.svg(720, 420);
    const names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'];
    const subs = ['bords', 'textures', 'motifs', 'parties', 'objets', 'spécifique'];
    const blocks = names.map((n, i) => block(h, s, 40 + i * 94, 110, 80, 90, n));
    const head = block(h, s, 40 + 6 * 94, 110, 80, 90, 'Tête');
    const locks = names.map((n, i) => lock(h, s, 40 + i * 94 + 40, 118));
    subs.forEach((t, i) => text(h, s, 80 + i * 94, 226, t, { class: 'muted' }));
    text(h, s, 644, 226, 'tâche', { class: 'muted' });
    const caption = h.label(s, 40, 262, 640, 140, '', '');

    let k = 2;
    /**
     * Updates the blocks, padlocks and explanation for the current number of frozen layers.
     */
    const apply = () => {
      blocks.forEach((b, i) => {
        paint(b, i < k ? 'frozen' : 'train');
        locks[i].setAttribute('opacity', i < k ? 1 : 0);
      });
      paint(head, 'train');
      let msg;
      if (k === 0) {
        msg = `<b>Aucune couche gelée</b> : tout le réseau est réentraîné. Il faut beaucoup de données, sinon le modèle risque le sur-apprentissage.`;
      } else if (k <= 2) {
        msg = `<b>${k} couche(s) gelée(s)</b> : les couches génériques (bords, textures) sont conservées, le reste s'adapte à la nouvelle tâche. Compromis adapté à des données de taille moyenne.`;
      } else if (k <= 4) {
        msg = `<b>${k} couches gelées</b> : moins de paramètres à apprendre, ce qui convient quand les données sont peu nombreuses.`;
      } else {
        msg = `<b>${k} couches gelées</b> : seule la tête est apprise. Très peu de données suffisent, mais le modèle s'adapte peu à une tâche très différente.`;
      }
      caption.innerHTML = msg;
      h.typeset(caption);
    };

    h.slider({ label: 'Couches gelées', min: 0, max: 6, step: 1, value: k, decimals: 0, onInput: (v) => { k = v; apply(); } });
    apply();
    blocks.forEach((b) => appear(h, b.g, 300));
  }

  Course.register({
    id: 'transfer',
    order: 9,
    title: 'Pré-apprentissage & finetuning',
    summary: 'Pré-apprentissage non supervisé avec des auto-encodeurs, puis finetuning et transfert de modèles pré-entraînés.',
    pdfPages: '33',
    steps: [
      {
        title: 'Supervisé vs non supervisé',
        text: `<p>L'apprentissage <b>supervisé</b> utilise des exemples étiquetés. L'apprentissage <b>non supervisé</b> n'utilise que des entrées. Erhan, Bengio et al. (2010) ont montré que l'apprentissage non supervisé des couches permet de les <b>pré-entraîner</b>, ce qui facilite ensuite l'apprentissage supervisé des réseaux profonds.</p><p>Une façon d'apprendre sans étiquettes est l'<b>auto-encodeur</b>, présenté dans l'étape suivante.</p>`,
        check: {
          q: 'Quelle est la différence principale entre apprentissage supervisé et non supervisé ?',
          choices: ['Le non supervisé n\'utilise pas d\'étiquettes', 'Le non supervisé est toujours plus précis', 'Le supervisé n\'utilise pas de données'],
          answer: 0,
          explain: 'Sans étiquettes, le modèle doit découvrir seul la structure des données.'
        },
        draw: drawSupervised
      },
      {
        title: 'Auto-encodeur',
        text: `<p>Un <b>auto-encodeur</b> est un réseau qui apprend à se reconstruire lui-même. L'<b>encodeur</b> compresse l'entrée en un <b>code</b> de petite dimension (le goulot d'étranglement), et le <b>décodeur</b> tente de reconstruire l'entrée à partir de ce code.</p><p>On minimise l'<b>erreur de reconstruction</b> : plus elle est faible, plus le code conserve l'information essentielle. Aucune étiquette n'est nécessaire.</p>`,
        draw: drawAutoencoder
      },
      {
        title: 'Pré-apprentissage glouton',
        text: `<p>Le pré-apprentissage se fait <b>couche par couche</b> (glouton) : on entraîne un premier auto-encodeur sur les données, on gèle cette couche, puis on entraîne la suivante sur ses sorties, et ainsi de suite.</p><p>Une fois toutes les couches pré-entraînées, on les <b>empile</b>, on ajoute une sortie, et on affine l'ensemble avec les étiquettes.</p>`,
        draw: drawGreedy
      },
      {
        title: 'Finetuning',
        text: `<p>Yosinski et al. (2014) ont étudié la transférabilité des caractéristiques : on part d'un <b>modèle pré-entraîné</b> sur une autre tâche et/ou un autre jeu de données, puis on <b>affine</b> certaines couches et on en <b>apprend</b> de nouvelles.</p><p>Le bouton <b>Geler</b> verrouille les couches pré-entraînées (cadenas) : seule la nouvelle couche apprend. Le bouton <b>Affiner tout</b> ajuste aussi les couches existantes.</p>`,
        check: {
          q: 'Que signifie « geler » une couche lors du finetuning ?',
          choices: ['Ses poids ne sont plus mis à jour pendant l\'entraînement', 'Sa taille est réduite', 'Elle est supprimée du réseau'],
          answer: 0,
          explain: 'Une couche gelée garde les poids appris ailleurs ; seules les couches non gelées sont ajustées.'
        },
        draw: drawFinetune
      },
      {
        title: 'Transfert et couches gelées',
        text: `<p>Dans un réseau profond, les <b>premières couches</b> apprennent des caractéristiques <b>génériques</b> (bords, textures), utiles pour beaucoup de tâches. Les <b>dernières couches</b> sont plus <b>spécifiques</b> à la tâche.</p><p>Le curseur choisit combien de couches geler. Si le jeu de données est <b>petit</b>, on gèle davantage pour éviter le sur-apprentissage ; s'il est grand et que la tâche diffère, on en gèle moins.</p>`,
        check: {
          q: 'Pourquoi geler les premières couches lorsque le jeu de données est petit ?',
          choices: ['Elles apprennent des caractéristiques génériques, et on évite de sur-apprendre', 'Elles sont toujours inutiles', 'Pour rendre le calcul plus lent'],
          answer: 0,
          explain: 'Les premières couches sont peu spécifiques : les réentraîner avec peu de données favorise le sur-apprentissage.'
        },
        draw: drawSummary
      }
    ]
  });
})();
