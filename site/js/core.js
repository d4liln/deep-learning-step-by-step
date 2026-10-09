/**
 * Shared course registry and drawing toolkit used by every lesson file.
 * Lessons call Course.register(...) and receive a fresh toolkit (h) on each step draw.
 */
(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const Course = {
    lessons: [],
    quiz: [],

    /**
     * Registers a lesson. See README section "Lesson contract" for the full schema.
     */
    register(lesson) {
      if (!lesson || !lesson.id || !Array.isArray(lesson.steps)) {
        console.error('Invalid lesson', lesson);
        return;
      }
      this.lessons.push(lesson);
      this.lessons.sort((a, b) => (a.order || 0) - (b.order || 0));
    },

    /**
     * Appends questions to the global quiz bank.
     */
    addQuestions(list) {
      this.quiz.push(...list);
    },

    /**
     * Builds the toolkit handed to a step's draw function. Every timer, loop and listener
     * created through it is cancelled automatically when the step is left.
     */
    createToolkit(stage, controls, shared) {
      let alive = true;
      const rafs = new Set();
      const timers = new Set();
      const cleanups = [];

      const css = getComputedStyle(document.documentElement);

      const h = {
        stage,
        controls,
        shared,

        /** Returns false once the step has been left; long async sequences must check it. */
        alive: () => alive,

        /** Reads a theme colour token (pos, neg, accent, accent-2, ink, muted, line, surface, surface-2, ok, bad, warn). */
        color(name) {
          return css.getPropertyValue('--' + name).trim() || '#888';
        },

        /** Creates a responsive SVG with the given viewBox size inside parent (default: stage). */
        svg(width, height, parent) {
          const s = document.createElementNS(SVG_NS, 'svg');
          s.setAttribute('viewBox', `0 0 ${width} ${height}`);
          s.setAttribute('class', 'viz');
          s.setAttribute('preserveAspectRatio', 'xMidYMid meet');
          (parent || stage).appendChild(s);
          h.defs(s);
          return s;
        },

        /** Adds arrow markers (arrow, arrow-accent, arrow-pos, arrow-neg) to an SVG. */
        defs(s) {
          const d = document.createElementNS(SVG_NS, 'defs');
          [['arrow', 'ink'], ['arrow-accent', 'accent'], ['arrow-pos', 'pos'], ['arrow-neg', 'neg'], ['arrow-muted', 'muted']]
            .forEach(([id, c]) => {
              const m = document.createElementNS(SVG_NS, 'marker');
              m.setAttribute('id', id);
              m.setAttribute('viewBox', '0 0 10 10');
              m.setAttribute('refX', '9');
              m.setAttribute('refY', '5');
              m.setAttribute('markerWidth', '7');
              m.setAttribute('markerHeight', '7');
              m.setAttribute('orient', 'auto-start-reverse');
              const p = document.createElementNS(SVG_NS, 'path');
              p.setAttribute('d', 'M0,0 L10,5 L0,10 z');
              p.setAttribute('fill', `var(--${c})`);
              m.appendChild(p);
              d.appendChild(m);
            });
          s.appendChild(d);
          return d;
        },

        /**
         * Creates an SVG element. attrs.text sets textContent; attrs.class sets the class.
         * Returns the element.
         */
        el(tag, attrs, parent) {
          const e = document.createElementNS(SVG_NS, tag);
          h.attr(e, attrs);
          if (parent) parent.appendChild(e);
          return e;
        },

        /** Sets attributes on an element (text -> textContent, html -> innerHTML). */
        attr(e, attrs) {
          if (!attrs) return e;
          for (const k in attrs) {
            const v = attrs[k];
            if (v === undefined || v === null) continue;
            if (k === 'text') e.textContent = v;
            else if (k === 'html') e.innerHTML = v;
            else e.setAttribute(k, v);
          }
          return e;
        },

        /** Creates an HTML element with attributes, appended to parent when given. */
        html(tag, attrs, parent) {
          const e = document.createElement(tag);
          h.attr(e, attrs);
          if (parent) parent.appendChild(e);
          return e;
        },

        /** Renders a TeX string into an element (KaTeX when available, plain text otherwise). */
        tex(str, parent, display) {
          const span = document.createElement(display ? 'div' : 'span');
          span.className = 'tex';
          if (window.katex) {
            try { window.katex.render(str, span, { displayMode: !!display, throwOnError: false }); }
            catch (e) { span.textContent = str; }
          } else span.textContent = str;
          if (parent) parent.appendChild(span);
          return span;
        },

        /** Places HTML (including $TeX$) inside an SVG at x,y via foreignObject. */
        label(svg, x, y, w, hgt, content, cls) {
          const fo = h.el('foreignObject', { x, y, width: w, height: hgt }, svg);
          const div = document.createElement('div');
          div.className = 'fo-label ' + (cls || '');
          div.innerHTML = content;
          fo.appendChild(div);
          h.typeset(div);
          return div;
        },

        /** Typesets $...$ and $$...$$ in an HTML element. */
        typeset(node) {
          if (window.renderMathInElement) {
            window.renderMathInElement(node, {
              delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
              ],
              throwOnError: false
            });
          }
        },

        /** Runs fn(elapsedMs, dtMs) every frame until it returns false or the step is left. */
        loop(fn) {
          let start = null, last = null, id;
          const tick = (t) => {
            if (!alive) return;
            if (start === null) { start = t; last = t; }
            const keep = fn(t - start, t - last);
            last = t;
            if (keep === false) { rafs.delete(id); return; }
            id = requestAnimationFrame(tick);
            rafs.add(id);
          };
          id = requestAnimationFrame(tick);
          rafs.add(id);
        },

        /** Calls fn after ms, unless the step was left. */
        after(ms, fn) {
          const id = setTimeout(() => { timers.delete(id); if (alive) fn(); }, ms);
          timers.add(id);
          return id;
        },

        /** Calls fn every ms until the step is left. */
        every(ms, fn) {
          const id = setInterval(() => { if (alive) fn(); }, ms);
          timers.add(id);
          return id;
        },

        /** Promise resolved after ms; never resolves once the step is left (sequences stop silently). */
        sleep(ms) {
          return new Promise((res) => h.after(ms, res));
        },

        /** Animates p from 0 to 1 over ms with easing, calling fn(p). Resolves when done. */
        tween(ms, fn, ease) {
          const e = ease || h.ease.inOut;
          return new Promise((res) => {
            h.loop((t) => {
              const p = Math.min(1, t / ms);
              fn(e(p));
              if (p >= 1) { res(); return false; }
            });
          });
        },

        ease: {
          linear: (p) => p,
          inOut: (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2),
          out: (p) => 1 - Math.pow(1 - p, 3)
        },

        /** Linear interpolation. */
        lerp: (a, b, p) => a + (b - a) * p,

        /** Returns a linear scale function mapping [d0,d1] to [r0,r1]. */
        scale(d0, d1, r0, r1) {
          const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
          f.invert = (v) => d0 + ((v - r0) / (r1 - r0)) * (d1 - d0);
          return f;
        },

        /** Formats a number with d decimals, using a French decimal comma. */
        fmt(x, d) {
          const n = d === undefined ? 2 : d;
          if (!isFinite(x)) return '—';
          return Number(x).toFixed(n).replace('.', ',');
        },

        /** Seeded pseudo-random generator so that every replay looks identical. */
        rng(seed) {
          let s = seed >>> 0 || 1;
          return () => {
            s ^= s << 13; s >>>= 0;
            s ^= s >> 17;
            s ^= s << 5; s >>>= 0;
            return s / 4294967296;
          };
        },

        /** Adds a labelled range slider to the controls bar; onInput(value) is called on change. */
        slider(opts) {
          const wrap = h.html('label', { class: 'ctl ctl-slider' }, controls);
          const name = h.html('span', { class: 'ctl-name', html: opts.label }, wrap);
          h.typeset(name);
          const input = h.html('input', {
            type: 'range', min: opts.min, max: opts.max, step: opts.step || 0.01, value: opts.value
          }, wrap);
          const out = h.html('output', { class: 'ctl-val' }, wrap);
          const show = (v) => { out.textContent = opts.format ? opts.format(v) : h.fmt(v, opts.decimals === undefined ? 2 : opts.decimals); };
          show(+opts.value);
          const on = () => { const v = +input.value; show(v); if (opts.onInput) opts.onInput(v); };
          input.addEventListener('input', on);
          return { input, set(v) { input.value = v; show(+v); }, get: () => +input.value };
        },

        /** Adds a button to the controls bar. */
        button(label, onClick, cls) {
          const b = h.html('button', { type: 'button', class: 'ctl ctl-btn ' + (cls || ''), html: label }, controls);
          b.addEventListener('click', onClick);
          return b;
        },

        /** Adds a segmented toggle (list of [value,label]) to the controls bar. */
        toggle(options, value, onChange) {
          const wrap = h.html('div', { class: 'ctl ctl-toggle', role: 'group' }, controls);
          const btns = options.map(([v, l]) => {
            const b = h.html('button', { type: 'button', html: l, 'aria-pressed': String(v === value) }, wrap);
            b.addEventListener('click', () => {
              btns.forEach((x) => x.setAttribute('aria-pressed', 'false'));
              b.setAttribute('aria-pressed', 'true');
              onChange(v);
            });
            return b;
          });
          return wrap;
        },

        /** Adds a small read-out panel to the controls bar; returns a setter taking HTML. */
        readout(initial) {
          const d = h.html('div', { class: 'ctl ctl-readout', html: initial || '' }, controls);
          h.typeset(d);
          return (content) => { d.innerHTML = content; h.typeset(d); };
        },

        /** Registers a cleanup callback run when the step is left. */
        onLeave(fn) { cleanups.push(fn); },

        /** Internal: stops everything created through this toolkit. */
        _dispose() {
          alive = false;
          rafs.forEach((id) => cancelAnimationFrame(id));
          timers.forEach((id) => { clearTimeout(id); clearInterval(id); });
          cleanups.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
        }
      };
      return h;
    }
  };

  window.Course = Course;
})();
