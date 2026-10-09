/**
 * Lesson: introduction to machine learning and deep learning, short history and first applications.
 */
(function () {
  /**
   * Adds a rounded box with HTML content centred inside, hidden until animated. Returns its group.
   */
  function box(h, parent, x, y, w, hh, html, stroke) {
    const g = h.el('g', { opacity: 0 }, parent);
    h.el('rect', { x, y, width: w, height: hh, rx: 10, fill: 'var(--surface)', stroke: stroke || 'var(--line)', 'stroke-width': 1.5 }, g);
    h.label(g, x + 6, y + 6, w - 12, hh - 12, html, 'center');
    return g;
  }

  /**
   * Adds an arrow line, hidden until animated.
   */
  function arrow(h, parent, x1, y1, x2, y2) {
    return h.el('line', { x1, y1, x2, y2, stroke: 'var(--muted)', 'stroke-width': 1.8, 'marker-end': 'url(#arrow)', opacity: 0 }, parent);
  }

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
   * Moves a dot from one point to another, then removes it.
   */
  function flow(h, parent, x1, y1, x2, y2, ms) {
    const dot = h.el('circle', { r: 5, fill: 'var(--accent)', cx: x1, cy: y1 }, parent);
    return h.tween(ms, (p) => h.attr(dot, { cx: h.lerp(x1, x2, p), cy: h.lerp(y1, y2, p) }))
      .then(() => dot.remove());
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
   * Paints a layer block in one of the states train, frozen or todo.
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
   * Draws the machine learning scheme: labelled examples train a model, which then predicts a new example.
   */
  function drawMl(h) {
    const s = h.svg(720, 420);
    const ys = [40, 150, 260];
    const ends = [160, 190, 220];
    const cards = [];
    const ins = [];
    ys.forEach((y, i) => {
      const animal = i === 1 ? 'chien' : 'chat';
      cards.push(box(h, s, 30, y, 235, 80, `<b>exemple ${i + 1}</b><br>entrée : image de ${animal}<br>étiquette : « ${animal} »`));
      ins.push(arrow(h, s, 265, y + 40, 300, ends[i]));
    });
    const model = box(h, s, 300, 140, 120, 100, `<b>modèle</b><br>$f(x) \\approx y$`, 'var(--accent)');
    const newCard = box(h, s, 500, 40, 190, 80, `<b>nouvel exemple</b><br>entrée : image ?`, 'var(--accent)');
    const newArrow = arrow(h, s, 500, 100, 420, 170);
    const out = box(h, s, 500, 250, 190, 90, `<b>prédiction</b><br>« chat » (95 %)`, 'var(--ok)');
    const outArrow = arrow(h, s, 420, 210, 500, 285);
    text(h, s, 360, 392, 'Paires (entrée, étiquette) → modèle → prédiction sur un nouvel exemple', { class: 'muted' });

    const run = async () => {
      appear(h, model, 400);
      for (let i = 0; i < 3; i++) {
        await appear(h, cards[i], 300);
        appear(h, ins[i], 200);
        await flow(h, s, 265, ys[i] + 40, 300, ends[i], 450);
        await h.sleep(100);
      }
      await appear(h, newCard, 300);
      appear(h, newArrow, 200);
      await flow(h, s, 500, 100, 420, 170, 500);
      appear(h, outArrow, 200);
      await flow(h, s, 420, 210, 500, 285, 500);
      await appear(h, out, 400);
    };
    run();
  }

  /**
   * Draws the data versus performance curves, with the deep learning curve overtaking classical methods.
   */
  function drawBigData(h) {
    const s = h.svg(720, 420);
    const X = h.scale(0, 10, 90, 660);
    const Y = h.scale(0, 1, 340, 70);
    const classic = (x) => 0.3 + 0.4 * (1 - Math.exp(-x / 2.5));
    const deep = (x) => 0.1 + 0.9 * (1 - Math.exp(-x / 6));
    h.el('line', { x1: 90, y1: 340, x2: 670, y2: 340, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, s);
    h.el('line', { x1: 90, y1: 340, x2: 90, y2: 60, stroke: 'var(--ink)', 'stroke-width': 1.5, 'marker-end': 'url(#arrow)' }, s);
    text(h, s, 380, 378, 'quantité de données');
    text(h, s, 40, 200, 'performance', { transform: 'rotate(-90 40 200)' });

    const bars = [];
    for (let i = 0; i <= 10; i++) {
      const hgt = 8 + i * 22;
      bars.push({ el: h.el('rect', { x: X(i) - 13, y: 340, width: 26, height: 0, fill: 'var(--muted)', opacity: 0.22 }, s), hgt });
    }
    const cl = h.el('polyline', { fill: 'none', stroke: 'var(--muted)', 'stroke-width': 3, 'stroke-linejoin': 'round' }, s);
    const dp = h.el('polyline', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 3.5, 'stroke-linejoin': 'round' }, s);
    const dot = h.el('circle', { r: 5, fill: 'var(--accent)' }, s);

    h.el('line', { x1: 100, y1: 76, x2: 130, y2: 76, stroke: 'var(--muted)', 'stroke-width': 3 }, s);
    text(h, s, 138, 80, 'méthodes classiques', { 'text-anchor': 'start' });
    h.el('line', { x1: 100, y1: 100, x2: 130, y2: 100, stroke: 'var(--accent)', 'stroke-width': 3.5 }, s);
    text(h, s, 138, 104, 'deep learning', { 'text-anchor': 'start' });

    let xc = 10;
    for (let x = 0; x <= 10; x += 0.05) {
      if (deep(x) >= classic(x)) { xc = x; break; }
    }

    const points = (fn, xmax) => {
      const out = [];
      for (let k = 0; k <= 100; k++) {
        const x = (xmax * k) / 100;
        out.push(X(x).toFixed(1) + ',' + Y(fn(x)).toFixed(1));
      }
      return out.join(' ');
    };

    const mark = h.el('circle', { cx: X(xc), cy: Y(deep(xc)), r: 8, fill: 'none', stroke: 'var(--accent-2)', 'stroke-width': 2.5, opacity: 0 }, s);
    const note = box(h, s, 380, 56, 300, 46, `Au-delà d'un certain volume, le deep learning dépasse les méthodes classiques`);

    const run = async () => {
      await h.tween(3200, (p) => {
        const xmax = 10 * p;
        bars.forEach((b, i) => {
          const frac = Math.min(1, Math.max(0, xmax - i + 1));
          b.el.setAttribute('height', b.hgt * frac);
          b.el.setAttribute('y', 340 - b.hgt * frac);
        });
        cl.setAttribute('points', points(classic, xmax));
        dp.setAttribute('points', points(deep, xmax));
        h.attr(dot, { cx: X(xmax), cy: Y(deep(xmax)) });
      }, h.ease.linear);
      dot.setAttribute('opacity', 0);
      await appear(h, mark, 400);
      await appear(h, note, 400);
    };
    run();
  }

  /**
   * Draws a stack of layers turning raw pixels into an object, with a signal crossing the layers.
   */
  function drawDeep(h) {
    const s = h.svg(720, 420);
    const names = ['pixels', 'bords et textures', 'motifs et parties', 'objet'];
    const groups = [];
    const rects = [];
    const arrows = [];
    for (let i = 0; i < 4; i++) {
      const x = 50 + i * 165;
      const g = h.el('g', { opacity: 0 }, s);
      rects.push(h.el('rect', { x, y: 80, width: 120, height: 220, rx: 12, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 2.5 }, g));
      for (let c = 0; c < 3; c++) {
        for (let k = 0; k < 5; k++) {
          h.el('circle', { cx: x + 30 + c * 30, cy: 110 + k * 40, r: 8, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.2 }, g);
        }
      }
      text(h, g, x + 60, 326, names[i], { 'font-weight': 600 });
      text(h, g, x + 60, 70, `couche ${i + 1}`, { class: 'muted' });
      groups.push(g);
      if (i < 3) arrows.push(arrow(h, s, x + 125, 190, x + 160, 190));
    }
    text(h, s, 360, 392, 'Chaque couche transforme la représentation de la précédente', { class: 'muted' });

    const run = async () => {
      for (let i = 0; i < 4; i++) {
        await appear(h, groups[i], 350);
        if (i < 3) appear(h, arrows[i], 250);
      }
      await h.sleep(300);
      rects[0].setAttribute('stroke', 'var(--accent)');
      for (let i = 0; i < 3; i++) {
        await flow(h, s, 110 + i * 165, 190, 110 + (i + 1) * 165, 190, 600);
        rects[i + 1].setAttribute('stroke', 'var(--accent)');
      }
    };
    run();
  }

  const EVENTS = [
    { year: '1957', name: 'Perceptron', who: 'Rosenblatt', desc: 'Premier modèle apprenant à partir d\'exemples : un neurone linéaire dont les poids sont ajustés pour classer les données.' },
    { year: '1969', name: 'Problème XOR', who: 'Minsky, Papert', desc: 'Un perceptron seul ne peut pas représenter le XOR. L\'intérêt pour les réseaux de neurones recule pendant plusieurs années.' },
    { year: '1986', name: 'MLP et rétropropagation', who: 'Rumelhart et al.', desc: 'Des couches cachées et la rétropropagation du gradient permettent d\'entraîner des perceptrons multicouches.' },
    { year: '1992', name: 'SVM', who: 'Vapnik et al.', desc: 'Machines à vecteurs de support : méthode à marge maximale, très performante pendant les années 1990 et 2000.' },
    { year: '1998', name: 'ConvNets', who: 'LeCun et al.', desc: 'Réseaux convolutifs appliqués à la reconnaissance de chiffres manuscrits, avec partage des poids entre positions.' },
    { year: '2010', name: 'Deep Neural Networks', who: 'Hinton et al.', desc: 'Entraînement efficace de réseaux profonds grâce à de meilleures techniques d\'apprentissage et au calcul sur GPU.' },
    { year: '2012', name: 'AlexNet', who: 'Krizhevsky, Hinton et al.', desc: '8 couches : victoire écrasante au défi ImageNet, point de bascule du deep learning en vision.' },
    { year: '2014', name: 'GoogLeNet', who: 'Google', desc: '22 couches grâce à des modules « inception » qui restent raisonnables en calcul.' },
    { year: '2016', name: 'ResNet', who: 'He et al.', desc: '152 couches : les connexions résiduelles facilitent l\'entraînement de réseaux très profonds.' },
    { year: '2017', name: 'AlphaGo', who: 'DeepMind', desc: 'Combine apprentissage par renforcement et réseaux convolutifs pour battre des champions du jeu de Go.' }
  ];

  const LAYERS = [['AlexNet (2012)', 8], ['GoogLeNet (2014)', 22], ['ResNet (2016)', 152]];

  /**
   * Draws the interactive timeline of the main milestones, with a bar chart of network depths.
   */
  function drawHistory(h) {
    const s = h.svg(720, 420);
    const axisY = 150;
    h.el('line', { x1: 40, y1: axisY, x2: 700, y2: axisY, stroke: 'var(--ink)', 'stroke-width': 2, 'marker-end': 'url(#arrow)' }, s);
    let sel = -1;
    const desc = h.label(s, 40, 226, 640, 66, 'Cliquez sur une date, ou utilisez les boutons ◀ ▶.', '');

    const select = (i) => {
      sel = i;
      nodes.forEach((n, k) => {
        n.circle.setAttribute('fill', k === i ? 'var(--accent)' : 'var(--surface)');
        n.circle.setAttribute('stroke', k === i ? 'var(--accent)' : 'var(--ink)');
      });
      const ev = EVENTS[i];
      if (!ev) return;
      desc.innerHTML = `<b>${ev.year} : ${ev.name}</b> (${ev.who})<br>${ev.desc}`;
      h.typeset(desc);
    };

    const nodes = EVENTS.map((ev, i) => {
      const x = 60 + i * 66;
      const up = i % 2 === 0;
      const g = h.el('g', { opacity: 0, style: 'cursor: pointer' }, s);
      h.el('line', { x1: x, y1: up ? axisY - 10 : axisY + 10, x2: x, y2: up ? axisY - 26 : axisY + 26, stroke: 'var(--line)', 'stroke-width': 1.5 }, g);
      const circle = h.el('circle', { cx: x, cy: axisY, r: 8, fill: 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 2 }, g);
      text(h, g, x, up ? 98 : 180, ev.year, { 'font-weight': 700 });
      text(h, g, x, up ? 114 : 196, ev.name, { class: 'muted', 'font-size': 11 });
      g.addEventListener('click', () => select(i));
      return { g, circle };
    });

    h.button('◀', () => select(Math.max(0, sel - 1)));
    h.button('▶', () => select(Math.min(EVENTS.length - 1, sel + 1)));

    text(h, s, 40, 300, 'Nombre de couches', { 'text-anchor': 'start', 'font-weight': 700 });
    const bars = LAYERS.map(([label, n], i) => {
      const y = 316 + i * 30;
      h.el('text', { x: 40, y: y + 14, text: label, class: 'muted' }, s);
      const r = h.el('rect', { x: 200, y, width: 0, height: 20, rx: 4, fill: 'var(--accent)' }, s);
      const v = h.el('text', { x: 208, y: y + 14, text: String(n), 'font-weight': 700, 'text-anchor': 'start' }, s);
      return { r, v, w: n * 2.8 };
    });

    const run = async () => {
      for (const n of nodes) {
        await appear(h, n.g, 250);
        await h.sleep(120);
      }
      await h.tween(900, (p) => {
        bars.forEach((b) => {
          b.r.setAttribute('width', b.w * p);
          b.v.setAttribute('x', 208 + b.w * p);
        });
      });
    };
    select(-1);
    run();
  }

  /**
   * Draws the main successes of deep learning and the recent timeline up to chatbots.
   */
  function drawSummary(h) {
    const s = h.svg(720, 420);
    const top = box(h, s, 240, 30, 240, 56, `<b>Deep learning</b><br>représentations apprises`, 'var(--accent)');
    const branches = [
      ['Vision', 'reconnaissance, détection, segmentation'],
      ['Langage (NLP)', 'traduction, génération de texte'],
      ['IA : jeux et décision', 'apprentissage par renforcement'],
      ['Données structurées', 'spatiales et/ou temporelles']
    ];
    const bGroups = branches.map(([t, d], i) => {
      const x = 35 + i * 165;
      const line = h.el('line', { x1: 360, y1: 86, x2: x + 75, y2: 150, stroke: 'var(--muted)', 'stroke-width': 1.5, opacity: 0 }, s);
      const g = box(h, s, x, 150, 150, 90, `<b>${t}</b><br>${d}`);
      return { g, line };
    });
    h.el('line', { x1: 40, y1: 275, x2: 700, y2: 275, stroke: 'var(--ink)', 'stroke-width': 2, 'marker-end': 'url(#arrow)' }, s);
    const periods = [
      [35, `<b>Depuis mi-2010</b><br>Google, Meta, Amazon, Apple s'y intéressent`],
      [265, `<b>2017-2020</b><br>IA générative : language models, transformers, diffusion`],
      [495, `<b>Aujourd'hui</b><br>chatbots et VLM (modèles vision-langage)`]
    ];
    const pGroups = periods.map(([x, html], i) => {
      const c = h.el('circle', { cx: x + 95, cy: 275, r: 6, fill: 'var(--accent)', opacity: 0 }, s);
      const g = box(h, s, x, 300, 190, 90, html);
      return { g, c };
    });

    const run = async () => {
      await appear(h, top, 400);
      for (const b of bGroups) {
        appear(h, b.line, 300);
        await appear(h, b.g, 300);
        await h.sleep(120);
      }
      for (const p of pGroups) {
        await appear(h, p.c, 200);
        await appear(h, p.g, 300);
        await h.sleep(150);
      }
    };
    run();
  }

  /**
   * Draws three application panels: image classification, speech synthesis and the game of Go.
   */
  function drawApps(h) {
    const s = h.svg(720, 420);
    const panels = [[20, 'Vision'], [250, 'Synthèse de la parole'], [480, 'Jeux (Go)']];
    panels.forEach(([x, t]) => {
      h.el('rect', { x, y: 60, width: 220, height: 280, rx: 14, fill: 'var(--surface-2)', stroke: 'var(--line)', 'stroke-width': 1.5 }, s);
      text(h, s, x + 110, 46, t, { 'font-weight': 700 });
    });

    const rng = h.rng(11);
    const tiles = [];
    const fills = ['var(--line)', 'var(--muted)', 'var(--surface)'];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 4; c++) {
        tiles.push(h.el('rect', { x: 38 + c * 48, y: 88 + r * 48, width: 40, height: 40, rx: 4, fill: fills[Math.floor(rng() * 3)], opacity: 0.6, stroke: 'var(--line)', 'stroke-width': 1 }, s));
      }
    }
    const tileLabel = text(h, s, 130, 252, 'chat (96 %)', { 'font-weight': 700, style: 'fill: var(--accent)', opacity: 0 });
    text(h, s, 130, 290, '1 000 classes', { class: 'muted' });
    text(h, s, 130, 308, "parmi environ un million d'images", { class: 'muted' });
    text(h, s, 130, 326, 'détection, segmentation, génération', { class: 'muted' });

    const ox = 270, baseY = 190, span = 180, N = 180;
    const wave = h.el('polyline', { fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, s);
    h.el('line', { x1: ox, y1: baseY, x2: ox + span, y2: baseY, stroke: 'var(--line)', 'stroke-width': 1 }, s);
    text(h, s, 360, 290, 'WaveNet (DeepMind)', { 'font-weight': 700 });
    text(h, s, 360, 308, "forme d'onde brute,", { class: 'muted' });
    text(h, s, 360, 326, 'échantillon par échantillon', { class: 'muted' });
    h.loop((t) => {
      const ph = t / 300;
      const pts = [];
      for (let k = 0; k <= N; k++) {
        const x = k / N;
        const v = 0.6 * Math.sin(2 * Math.PI * 3 * x + ph)
          + 0.3 * Math.sin(2 * Math.PI * 9 * x - ph * 1.7)
          + 0.1 * Math.sin(2 * Math.PI * 21 * x + ph * 2.3);
        pts.push((ox + x * span).toFixed(1) + ',' + (baseY - 55 * v).toFixed(1));
      }
      wave.setAttribute('points', pts.join(' '));
    });

    const gx = 502, gy = 90, cell = 22;
    for (let i = 0; i < 9; i++) {
      h.el('line', { x1: gx, y1: gy + i * cell, x2: gx + 8 * cell, y2: gy + i * cell, stroke: 'var(--ink)', 'stroke-width': 1 }, s);
      h.el('line', { x1: gx + i * cell, y1: gy, x2: gx + i * cell, y2: gy + 8 * cell, stroke: 'var(--ink)', 'stroke-width': 1 }, s);
    }
    const moves = [[4, 4], [2, 2], [6, 2], [2, 6], [6, 6], [4, 2], [4, 6], [2, 4], [6, 4]];
    const stones = moves.map(([i, j], k) => {
      const black = k % 2 === 0;
      return h.el('circle', {
        cx: gx + i * cell, cy: gy + j * cell, r: 9, opacity: 0,
        fill: black ? 'var(--ink)' : 'var(--surface)', stroke: 'var(--ink)', 'stroke-width': 1.5
      }, s);
    });
    text(h, s, 590, 290, 'AlphaGo (DeepMind)', { 'font-weight': 700 });
    text(h, s, 590, 308, 'apprentissage par renforcement', { class: 'muted' });
    text(h, s, 590, 326, '+ réseaux convolutifs (CNN)', { class: 'muted' });

    const run = async () => {
      await h.sleep(300);
      tiles[6].setAttribute('stroke', 'var(--accent)');
      tiles[6].setAttribute('stroke-width', 3);
      await appear(h, tileLabel, 400);
      await h.sleep(400);
      for (const st of stones) {
        await appear(h, st, 250);
        await h.sleep(250);
      }
    };
    run();
  }

  Course.register({
    id: 'intro',
    order: 1,
    title: 'Introduction & historique',
    summary: 'Apprentissage automatique, deep learning, repères historiques et premières applications spectaculaires.',
    pdfPages: '1-10',
    steps: [
      {
        title: 'Apprentissage automatique',
        text: `<p>Certaines tâches sont impossibles à décrire par des règles écrites à la main : reconnaître un chat sur une photo, traduire une phrase. L'<b>apprentissage automatique</b> (machine learning) procède autrement : on montre au programme des <b>exemples</b>, et il apprend à résoudre la tâche seul.</p><p>En <b>apprentissage supervisé</b>, chaque exemple est une <b>entrée</b> associée à son <b>étiquette</b> (la bonne réponse). Le modèle ajuste ses paramètres pour reproduire ces étiquettes, puis prédit la sortie d'un exemple qu'il n'a jamais vu.</p>`,
        note: 'Une règle explicite n\'est pas nécessaire : le modèle la déduit des exemples.',
        check: {
          q: 'En apprentissage supervisé, que contient chaque exemple d\'entraînement ?',
          choices: ['Uniquement une entrée', 'Une entrée et son étiquette', 'Uniquement une étiquette'],
          answer: 1,
          explain: 'L\'étiquette fournit la bonne réponse, ce qui permet au modèle de corriger ses paramètres.'
        },
        draw: drawMl
      },
      {
        title: 'Big data',
        text: `<p>Le <b>big data</b> désigne des quantités de données très importantes. Plus on dispose d'exemples, plus un modèle à grande capacité peut exploiter ses paramètres.</p><p>Sur la figure, les <b>méthodes classiques</b> (en gris) atteignent vite un plateau : elles ne tirent plus parti des données supplémentaires. Le <b>deep learning</b> (en couleur) part moins bien avec peu de données, mais continue de progresser et finit par les dépasser.</p>`,
        draw: drawBigData
      },
      {
        title: 'Deep learning',
        text: `<p>Le <b>deep learning</b> est un modèle d'<b>apprentissage de représentations en couches</b>, étudié depuis les années 1950. Chaque couche transforme ce que reçoit la précédente : des pixels on passe aux bords, puis aux motifs, puis à l'objet.</p><p>Il existe de nombreuses variantes : apprentissage supervisé ou non supervisé, classification, régression, renforcement. Ces modèles demandent beaucoup de données et de paramètres ; ils s'accommodent du big data grâce à la <b>descente de gradient stochastique</b>, qui met à jour les paramètres sur de petits lots d'exemples.</p>`,
        draw: drawDeep
      },
      {
        title: 'Bref historique',
        text: `<p>Cliquez sur les dates pour lire ce qui a marqué chaque étape. Après le perceptron (1957) et le problème XOR (1969), la rétropropagation (1986) puis les réseaux convolutifs (1998) préparent la vague actuelle.</p><p>À partir de 2010, la profondeur explose : le nombre de couches passe de <b>8</b> (AlexNet, 2012) à <b>22</b> (GoogLeNet, 2014), puis à <b>152</b> (ResNet, 2016).</p>`,
        check: {
          q: 'Pourquoi le perceptron seul ne peut-il pas résoudre le problème XOR (1969) ?',
          choices: ['Il est trop lent', 'Il ne sait représenter que des frontières linéaires', 'Il manque de données'],
          answer: 1,
          explain: 'Le XOR n\'est pas séparable linéairement : il faut des couches cachées, d\'où l\'intérêt de la rétropropagation en 1986.'
        },
        draw: drawHistory
      },
      {
        title: 'Succès spectaculaires',
        text: `<p>Le deep learning a obtenu des <b>succès spectaculaires</b> en vision, en traitement du langage et en IA, notamment sur des données structurées dans l'espace ou dans le temps.</p><p>Depuis le milieu des années 2010, de grands acteurs privés (Google, Meta, Amazon, Apple) y investissent massivement. Depuis 2017-2020, l'<b>IA générative</b> (modèles de langage, réseaux adversariaux, modèles de diffusion, transformers) s'impose. Aujourd'hui, l'IA se manifeste surtout par les <b>chatbots</b> et les <b>VLM</b> (modèles vision-langage).</p>`,
        draw: drawSummary
      },
      {
        title: 'Applications',
        text: `<p><b>Vision</b> : sur ImageNet, il faut distinguer 1 000 classes parmi environ un million d'images. Les réseaux profonds ont fait chuter les erreurs de façon spectaculaire, et ouvert la détection d'objets, la segmentation ou la génération d'images.</p><p><b>Synthèse de la parole</b> : WaveNet (DeepMind) génère directement la forme d'onde audio, échantillon par échantillon.</p><p><b>Jeux</b> : AlphaGo combine apprentissage par renforcement et réseaux convolutifs (CNN) pour jouer au jeu de Go.</p>`,
        check: {
          q: 'Quel est le principal ingrédient d\'AlphaGo ?',
          choices: ['Une base de règles écrites à la main', 'Apprentissage par renforcement combiné à des réseaux convolutifs', 'Un simple perceptron'],
          answer: 1,
          explain: 'AlphaGo associe l\'apprentissage par renforcement à des CNN qui évaluent les positions et les coups.'
        },
        draw: drawApps
      }
    ]
  });
})();
