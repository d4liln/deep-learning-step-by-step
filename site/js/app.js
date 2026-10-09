/**
 * Application shell: routing, sidebar, step-by-step lesson player and quiz engine.
 */
(function () {
  const C = window.Course;
  const main = document.getElementById('main');
  const toc = document.getElementById('toc');
  const topbarTitle = document.getElementById('topbarTitle');

  /** Safe localStorage wrapper; the page works identically when storage is unavailable. */
  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
      catch (e) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage unavailable */ }
    }
  };

  let progress = store.get('dl-progress', {});
  let bestScores = store.get('dl-best', {});
  let current = null;

  /** Escapes text for safe HTML insertion. */
  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  /** Typesets TeX in a node when KaTeX auto-render is loaded. */
  function typeset(node) {
    if (window.renderMathInElement) {
      window.renderMathInElement(node, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  }

  /** Marks a lesson step as visited and refreshes the sidebar progress. */
  function markVisited(lessonId, stepIdx) {
    const list = progress[lessonId] || [];
    if (!list.includes(stepIdx)) {
      list.push(stepIdx);
      progress[lessonId] = list;
      store.set('dl-progress', progress);
      renderToc();
    }
  }

  /** Fraction of steps visited in a lesson. */
  function ratio(lesson) {
    const seen = (progress[lesson.id] || []).filter((i) => i < lesson.steps.length).length;
    return lesson.steps.length ? seen / lesson.steps.length : 0;
  }

  /** Renders the chapter list in the sidebar. */
  function renderToc() {
    const route = location.hash;
    toc.innerHTML = C.lessons.map((l, i) => {
      const r = ratio(l);
      const active = route.startsWith('#/lesson/' + l.id + '/') || route === '#/lesson/' + l.id;
      return `<a href="#/lesson/${l.id}/0" class="toc-item${active ? ' active' : ''}">
        <span class="toc-num">${String(i + 1).padStart(2, '0')}</span>
        <span class="toc-title">${esc(l.title)}</span>
        <span class="toc-prog" style="--p:${r}" title="${Math.round(r * 100)} % vu"></span>
      </a>`;
    }).join('');
  }

  /** Tears down the active lesson player (timers, animations, listeners). */
  function teardown() {
    if (current && current.toolkit) current.toolkit._dispose();
    if (current && current.autoplay) clearTimeout(current.autoplay);
    current = null;
  }

  /** Home page: hero and chapter cards. */
  function renderHome() {
    topbarTitle.textContent = 'Deep Learning pas à pas';
    const first = C.lessons[0];
    main.innerHTML = `
      <section class="home">
        <div class="hero">
          <p class="kicker">Apprentissage automatique avancé · S. Ayache · Polytech Marseille</p>
          <h1>Le Deep Learning,<br><em>une étape à la fois.</em></h1>
          <p class="lede">Chaque notion du cours est rejouée sous forme d'animation découpée en étapes.
          Avancez, reculez, sautez directement à une étape, manipulez les paramètres, puis vérifiez ce que vous avez retenu avec le quiz.</p>
          <div class="hero-actions">
            ${first ? `<a class="btn btn-primary" href="#/lesson/${first.id}/0">Commencer le cours</a>` : ''}
            <a class="btn" href="#/quiz">Aller au quiz</a>
          </div>
          <p class="hint">Raccourcis : <kbd>←</kbd> <kbd>→</kbd> étapes · <kbd>R</kbd> rejouer · <kbd>Espace</kbd> lecture auto</p>
        </div>
        <div class="cards">
          ${C.lessons.map((l, i) => {
            const r = ratio(l);
            const best = bestScores[l.id];
            return `<a class="card" href="#/lesson/${l.id}/0">
              <span class="card-num">${String(i + 1).padStart(2, '0')}</span>
              <h3>${esc(l.title)}</h3>
              <p>${l.summary || ''}</p>
              <div class="card-foot">
                <span class="bar"><span style="width:${Math.round(r * 100)}%"></span></span>
                <span>${l.steps.length} étapes${best !== undefined ? ` · quiz ${best} %` : ''}</span>
              </div>
            </a>`;
          }).join('')}
        </div>
      </section>`;
    typeset(main);
  }

  /** Lesson player for one lesson at a given step. */
  function renderLesson(id, stepIdx) {
    const lesson = C.lessons.find((l) => l.id === id);
    if (!lesson) { location.hash = '#/'; return; }
    const n = lesson.steps.length;
    const idx = Math.max(0, Math.min(n - 1, isNaN(stepIdx) ? 0 : stepIdx));
    const li = C.lessons.indexOf(lesson);
    const prevLesson = C.lessons[li - 1];
    const nextLesson = C.lessons[li + 1];

    const sameLesson = current && current.lesson === lesson;
    const shared = sameLesson ? current.shared : {};
    const wasPlaying = sameLesson && current.playing;
    if (!sameLesson && typeof lesson.init === 'function') lesson.init(shared);
    teardown();

    topbarTitle.textContent = lesson.title;
    const step = lesson.steps[idx];

    main.innerHTML = `
      <section class="lesson">
        <header class="lesson-head">
          <p class="kicker">Chapitre ${li + 1}${lesson.pdfPages ? ` · diapos ${esc(lesson.pdfPages)} du PDF` : ''}</p>
          <h1>${esc(lesson.title)}</h1>
        </header>
        <ol class="stepper" aria-label="Étapes">
          ${lesson.steps.map((s, i) => `
            <li><a href="#/lesson/${lesson.id}/${i}" class="${i === idx ? 'on' : ''}${(progress[lesson.id] || []).includes(i) ? ' seen' : ''}" title="${esc(s.title)}">
              <span class="dot">${i + 1}</span><span class="st">${esc(s.title)}</span>
            </a></li>`).join('')}
        </ol>
        <div class="lesson-body">
          <div class="viz-panel">
            <div class="stage" id="stage"></div>
            <div class="controls" id="controls"></div>
          </div>
          <article class="explain" id="explain">
            <p class="step-count">Étape ${idx + 1} / ${n}</p>
            <h2>${esc(step.title)}</h2>
            <div class="text">${step.text || ''}</div>
            ${step.note ? `<aside class="note"><strong>À retenir</strong>${step.note}</aside>` : ''}
            <div id="check"></div>
          </article>
        </div>
        <footer class="player">
          <div class="player-btns">
            <button type="button" class="pbtn" id="pFirst" title="Première étape" ${idx === 0 ? 'disabled' : ''}>⏮</button>
            <button type="button" class="pbtn" id="pPrev" title="Étape précédente (←)" ${idx === 0 ? 'disabled' : ''}>◀ Précédent</button>
            <button type="button" class="pbtn" id="pReplay" title="Rejouer l'animation (R)">↻ Rejouer</button>
            <button type="button" class="pbtn" id="pPlay" title="Lecture automatique (Espace)">${wasPlaying ? '❚❚ Pause' : '▶ Auto'}</button>
            <button type="button" class="pbtn pbtn-main" id="pNext" title="Étape suivante (→)" ${idx === n - 1 ? 'disabled' : ''}>Suivant ▶</button>
            <button type="button" class="pbtn" id="pLast" title="Dernière étape" ${idx === n - 1 ? 'disabled' : ''}>⏭</button>
          </div>
          <div class="progress"><span style="width:${((idx + 1) / n) * 100}%"></span></div>
        </footer>
        ${idx === n - 1 ? `
        <div class="end-card">
          <p>Fin du chapitre « ${esc(lesson.title)} ».</p>
          <div class="hero-actions">
            <a class="btn btn-primary" href="#/quiz/${lesson.id}">Me tester sur ce chapitre</a>
            ${nextLesson ? `<a class="btn" href="#/lesson/${nextLesson.id}/0">Chapitre suivant : ${esc(nextLesson.title)} →</a>` : ''}
          </div>
        </div>` : ''}
        <nav class="lesson-nav">
          ${prevLesson ? `<a href="#/lesson/${prevLesson.id}/0">← ${esc(prevLesson.title)}</a>` : '<span></span>'}
          ${nextLesson ? `<a href="#/lesson/${nextLesson.id}/0">${esc(nextLesson.title)} →</a>` : '<span></span>'}
        </nav>
      </section>`;

    const explain = document.getElementById('explain');
    typeset(explain);
    if (step.check) renderCheck(document.getElementById('check'), step.check);

    const go = (i) => { if (i >= 0 && i < n) location.hash = `#/lesson/${lesson.id}/${i}`; };
    current = { lesson, idx, shared, playing: wasPlaying, go };

    const draw = () => {
      if (current.toolkit) current.toolkit._dispose();
      const stage = document.getElementById('stage');
      const controls = document.getElementById('controls');
      stage.innerHTML = '';
      controls.innerHTML = '';
      const h = C.createToolkit(stage, controls, shared);
      current.toolkit = h;
      try {
        if (typeof step.draw === 'function') step.draw(h);
      } catch (e) {
        console.error(e);
        stage.innerHTML = `<p class="err">Erreur dans l'animation : ${esc(e.message)}</p>`;
      }
      controls.hidden = !controls.children.length;
    };
    current.replay = draw;
    draw();

    document.getElementById('pFirst').onclick = () => go(0);
    document.getElementById('pPrev').onclick = () => go(idx - 1);
    document.getElementById('pNext').onclick = () => go(idx + 1);
    document.getElementById('pLast').onclick = () => go(n - 1);
    document.getElementById('pReplay').onclick = draw;
    document.getElementById('pPlay').onclick = () => togglePlay();

    if (wasPlaying) scheduleAutoplay();
    markVisited(lesson.id, idx);

    const on = main.querySelector('.stepper a.on');
    if (on) on.scrollIntoView({ block: 'nearest', inline: 'center' });
  }

  /** Starts or stops automatic step advance. */
  function togglePlay() {
    if (!current) return;
    current.playing = !current.playing;
    const b = document.getElementById('pPlay');
    if (b) b.textContent = current.playing ? '❚❚ Pause' : '▶ Auto';
    if (current.playing) scheduleAutoplay();
    else clearTimeout(current.autoplay);
  }

  /** Schedules the next automatic advance using the step's duration. */
  function scheduleAutoplay() {
    const { lesson, idx } = current;
    clearTimeout(current.autoplay);
    if (idx >= lesson.steps.length - 1) {
      current.playing = false;
      const b = document.getElementById('pPlay');
      if (b) b.textContent = '▶ Auto';
      return;
    }
    const ms = lesson.steps[idx].duration || 7000;
    current.autoplay = setTimeout(() => current && current.playing && current.go(idx + 1), ms);
  }

  /** Inline self-check question attached to a step. */
  function renderCheck(host, check) {
    host.innerHTML = `<div class="check"><p class="check-q"><span class="tag">Question éclair</span>${check.q}</p>
      <div class="choices">${check.choices.map((c, i) => `<button type="button" class="choice" data-i="${i}">${c}</button>`).join('')}</div>
      <div class="check-fb"></div></div>`;
    typeset(host);
    host.querySelectorAll('.choice').forEach((b) => {
      b.addEventListener('click', () => {
        const i = +b.dataset.i;
        host.querySelectorAll('.choice').forEach((x) => {
          x.disabled = true;
          if (+x.dataset.i === check.answer) x.classList.add('right');
        });
        if (i !== check.answer) b.classList.add('wrong');
        const fb = host.querySelector('.check-fb');
        fb.innerHTML = `<p class="${i === check.answer ? 'ok' : 'ko'}">${i === check.answer ? 'Exact !' : 'Pas tout à fait.'}</p>${check.explain ? `<p>${check.explain}</p>` : ''}`;
        typeset(fb);
      });
    });
  }

  /** Fisher–Yates shuffle (returns a new array). */
  function shuffle(a) {
    const r = a.slice();
    for (let i = r.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [r[i], r[j]] = [r[j], r[i]];
    }
    return r;
  }

  /** Quiz setup screen; preselects one lesson when given. */
  function renderQuizSetup(preselect) {
    topbarTitle.textContent = 'Quiz';
    const counts = {};
    C.quiz.forEach((q) => { counts[q.lesson] = (counts[q.lesson] || 0) + 1; });
    const lessons = C.lessons.filter((l) => counts[l.id]);
    main.innerHTML = `
      <section class="quiz">
        <header class="lesson-head">
          <p class="kicker">Auto-évaluation · ${C.quiz.length} questions disponibles</p>
          <h1>Se tester</h1>
        </header>
        <div class="quiz-setup">
          <h3>Chapitres</h3>
          <div class="chips">
            ${lessons.map((l) => `<label class="chip"><input type="checkbox" value="${l.id}" ${!preselect || preselect === l.id ? 'checked' : ''}>
              <span>${esc(l.title)} <small>${counts[l.id]} q.${bestScores[l.id] !== undefined ? ` · meilleur ${bestScores[l.id]} %` : ''}</small></span></label>`).join('')}
          </div>
          <div class="chip-actions"><button type="button" class="linkbtn" id="qAll">Tout cocher</button><button type="button" class="linkbtn" id="qNone">Tout décocher</button></div>
          <h3>Nombre de questions</h3>
          <div class="ctl-toggle" id="qCount" role="group">
            ${[5, 10, 20, 0].map((v, i) => `<button type="button" data-v="${v}" aria-pressed="${i === 1}">${v || 'Toutes'}</button>`).join('')}
          </div>
          <button type="button" class="btn btn-primary" id="qStart">Lancer le quiz</button>
        </div>
      </section>`;
    let count = 10;
    main.querySelectorAll('#qCount button').forEach((b) => b.addEventListener('click', () => {
      main.querySelectorAll('#qCount button').forEach((x) => x.setAttribute('aria-pressed', 'false'));
      b.setAttribute('aria-pressed', 'true');
      count = +b.dataset.v;
    }));
    const boxes = () => [...main.querySelectorAll('.chips input')];
    document.getElementById('qAll').onclick = () => boxes().forEach((b) => { b.checked = true; });
    document.getElementById('qNone').onclick = () => boxes().forEach((b) => { b.checked = false; });
    document.getElementById('qStart').onclick = () => {
      const ids = boxes().filter((b) => b.checked).map((b) => b.value);
      if (!ids.length) return;
      let pool = shuffle(C.quiz.filter((q) => ids.includes(q.lesson)));
      if (count) pool = pool.slice(0, count);
      runQuiz(pool, ids);
    };
  }

  /** Checks a user's answer against a question definition. */
  function isCorrect(q, value) {
    if (q.type === 'multi') {
      const a = [...q.answer].sort().join(',');
      return [...value].sort().join(',') === a;
    }
    if (q.type === 'num') {
      const v = parseFloat(String(value).replace(',', '.'));
      return isFinite(v) && Math.abs(v - q.answer) <= (q.tol === undefined ? 0.01 : q.tol);
    }
    return value === q.answer;
  }

  /** Runs a quiz session over a list of questions. */
  function runQuiz(pool, ids) {
    let i = 0;
    const results = [];

    const show = () => {
      if (i >= pool.length) return finish();
      const q = pool[i];
      const lesson = C.lessons.find((l) => l.id === q.lesson);
      const choices = q.type === 'tf' ? ['Vrai', 'Faux'] : q.choices || [];
      main.innerHTML = `
        <section class="quiz">
          <div class="quiz-top"><span>Question ${i + 1} / ${pool.length}</span><span>${lesson ? esc(lesson.title) : ''}</span>
            <span>Score : ${results.filter((r) => r.ok).length}</span></div>
          <div class="progress"><span style="width:${(i / pool.length) * 100}%"></span></div>
          <div class="qcard">
            <p class="qtype">${q.type === 'multi' ? 'Plusieurs réponses possibles' : q.type === 'num' ? 'Réponse numérique' : q.type === 'tf' ? 'Vrai ou faux' : 'Une seule réponse'}</p>
            <div class="qtext">${q.q}</div>
            ${q.type === 'num'
              ? `<div class="numrow"><input type="text" inputmode="decimal" id="numIn" placeholder="Votre réponse"><button type="button" class="btn btn-primary" id="qValidate">Valider</button></div>`
              : `<div class="choices">${choices.map((c, k) => `<button type="button" class="choice" data-i="${k}" aria-pressed="false">${c}</button>`).join('')}</div>
                 ${q.type === 'multi' ? '<button type="button" class="btn btn-primary" id="qValidate">Valider</button>' : ''}`}
            <div class="qfb" id="qfb"></div>
          </div>
        </section>`;
      typeset(main);

      const fb = document.getElementById('qfb');
      const reveal = (value) => {
        const ok = isCorrect(q, q.type === 'tf' ? value === 0 : value);
        results.push({ q, ok });
        main.querySelectorAll('.choice').forEach((b) => {
          b.disabled = true;
          const k = +b.dataset.i;
          const good = q.type === 'multi' ? q.answer.includes(k) : q.type === 'tf' ? (k === 0) === q.answer : k === q.answer;
          if (good) b.classList.add('right');
          else if (q.type === 'multi' ? value.includes(k) : k === value) b.classList.add('wrong');
        });
        const v = document.getElementById('qValidate');
        if (v) v.remove();
        const numIn = document.getElementById('numIn');
        if (numIn) numIn.disabled = true;
        const target = lesson && q.step !== undefined ? `#/lesson/${lesson.id}/${q.step}` : lesson ? `#/lesson/${lesson.id}/0` : null;
        fb.innerHTML = `
          <p class="${ok ? 'ok' : 'ko'}">${ok ? 'Bonne réponse !' : 'Mauvaise réponse.'}${q.type === 'num' ? ` Réponse attendue : ${String(q.answer).replace('.', ',')}` : ''}</p>
          ${q.explain ? `<p>${q.explain}</p>` : ''}
          <div class="hero-actions">
            <button type="button" class="btn btn-primary" id="qNext">${i + 1 < pool.length ? 'Question suivante' : 'Voir le résultat'}</button>
            ${target ? `<a class="btn" href="${target}">Revoir l'animation</a>` : ''}
          </div>`;
        typeset(fb);
        const next = document.getElementById('qNext');
        next.focus();
        next.onclick = () => { i++; show(); };
      };

      if (q.type === 'num') {
        const input = document.getElementById('numIn');
        input.focus();
        const submit = () => { if (input.value.trim()) reveal(input.value); };
        document.getElementById('qValidate').onclick = submit;
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
      } else if (q.type === 'multi') {
        const picked = new Set();
        main.querySelectorAll('.choice').forEach((b) => b.addEventListener('click', () => {
          const k = +b.dataset.i;
          if (picked.has(k)) picked.delete(k); else picked.add(k);
          b.setAttribute('aria-pressed', String(picked.has(k)));
        }));
        document.getElementById('qValidate').onclick = () => reveal([...picked]);
      } else {
        main.querySelectorAll('.choice').forEach((b) => b.addEventListener('click', () => reveal(+b.dataset.i)));
      }
    };

    const finish = () => {
      const good = results.filter((r) => r.ok).length;
      const pct = pool.length ? Math.round((good / pool.length) * 100) : 0;
      const by = {};
      results.forEach((r) => {
        by[r.q.lesson] = by[r.q.lesson] || { ok: 0, n: 0 };
        by[r.q.lesson].n++;
        if (r.ok) by[r.q.lesson].ok++;
      });
      Object.keys(by).forEach((id) => {
        const p = Math.round((by[id].ok / by[id].n) * 100);
        if (by[id].n >= 3 && (bestScores[id] === undefined || p > bestScores[id])) bestScores[id] = p;
      });
      store.set('dl-best', bestScores);
      const wrong = results.filter((r) => !r.ok);
      main.innerHTML = `
        <section class="quiz">
          <header class="lesson-head"><p class="kicker">Résultat</p><h1>${good} / ${pool.length} <small>(${pct} %)</small></h1></header>
          <p class="lede">${pct === 100 ? 'Parfait, tout est maîtrisé.' : pct >= 70 ? 'Solide. Revoyez les quelques points manqués ci-dessous.' : 'Quelques notions à revoir : chaque erreur renvoie vers l\'animation correspondante.'}</p>
          <div class="breakdown">
            ${Object.keys(by).map((id) => {
              const l = C.lessons.find((x) => x.id === id);
              const p = by[id].ok / by[id].n;
              return `<div class="brow"><span>${l ? esc(l.title) : id}</span><span class="bar"><span style="width:${p * 100}%"></span></span><span>${by[id].ok}/${by[id].n}</span></div>`;
            }).join('')}
          </div>
          ${wrong.length ? `<h3>À revoir</h3><ol class="review">${wrong.map((r) => {
            const l = C.lessons.find((x) => x.id === r.q.lesson);
            const target = l ? `#/lesson/${l.id}/${r.q.step !== undefined ? r.q.step : 0}` : '#/';
            return `<li><div class="qtext">${r.q.q}</div>${r.q.explain ? `<p class="muted">${r.q.explain}</p>` : ''}<a href="${target}">Revoir l'animation →</a></li>`;
          }).join('')}</ol>` : ''}
          <div class="hero-actions">
            ${wrong.length ? '<button type="button" class="btn btn-primary" id="qRetry">Refaire uniquement les erreurs</button>' : ''}
            <button type="button" class="btn" id="qAgain">Nouveau quiz</button>
          </div>
        </section>`;
      typeset(main);
      const retry = document.getElementById('qRetry');
      if (retry) retry.onclick = () => runQuiz(shuffle(wrong.map((r) => r.q)), ids);
      document.getElementById('qAgain').onclick = () => renderQuizSetup(ids.length === 1 ? ids[0] : null);
    };

    show();
  }

  /** Dispatches the current hash to a view. */
  function route() {
    const parts = location.hash.replace(/^#\/?/, '').split('/');
    document.body.classList.remove('nav-open');
    if (parts[0] === 'lesson' && parts[1]) {
      renderLesson(parts[1], parseInt(parts[2] || '0', 10));
    } else {
      teardown();
      if (parts[0] === 'quiz') renderQuizSetup(parts[1] || null);
      else renderHome();
      window.scrollTo(0, 0);
    }
    renderToc();
  }

  document.addEventListener('keydown', (e) => {
    if (!current) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.key === 'ArrowRight') { current.go(current.idx + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { current.go(current.idx - 1); e.preventDefault(); }
    else if (e.key === 'Home') { current.go(0); e.preventDefault(); }
    else if (e.key === 'End') { current.go(current.lesson.steps.length - 1); e.preventDefault(); }
    else if (e.key === 'r' || e.key === 'R') current.replay();
    else if (e.key === ' ' && tag !== 'button') { togglePlay(); e.preventDefault(); }
  });

  document.getElementById('menuBtn').addEventListener('click', () => document.body.classList.toggle('nav-open'));

  const savedTheme = store.get('dl-theme', null);
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  document.getElementById('themeToggle').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    store.set('dl-theme', next);
    if (current) current.replay();
  });

  window.addEventListener('hashchange', route);
  window.addEventListener('load', () => {
    route();
  });
})();
