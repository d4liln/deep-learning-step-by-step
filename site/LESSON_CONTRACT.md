# Lesson contract (for contributors)

Static site, no build, opened by double-clicking `index.html` (file://). Plain scripts, **no ES modules, no imports, no external libs** (KaTeX is already loaded globally by the page).
Everything is in **French** (UI + content). Source course text: `../cours.txt` is NOT here; agents get the path in their prompt.

## File

Each lesson is ONE file `site/lessons/<id>.js`, wrapped in an IIFE, calling `Course.register({...})`:

```js
/**
 * Lesson: <short English summary>.
 */
(function () {
  Course.register({
    id: 'perceptron',            // must equal the file name
    order: 2,                    // sidebar order (given in your prompt)
    title: 'Neurone & perceptron',
    summary: 'Une phrase (HTML autorisé) affichée sur la carte d\'accueil.',
    pdfPages: '11-15',
    init(shared) { },            // optional: called once when the lesson is opened; put data shared by all steps in `shared`
    steps: [
      {
        title: 'Somme pondérée',                 // short (2-5 words), shown in the stepper
        text: '<p>HTML. TeX inline $a = \\sum_i w_i x_i$ or display $$...$$.</p>',
        note: 'Optional "À retenir" callout (HTML + TeX).',
        duration: 8000,                           // optional, ms before auto-advance in autoplay (default 7000)
        check: {                                  // optional inline quick question
          q: 'Question (HTML/TeX)', choices: ['A', 'B', 'C'], answer: 1, explain: 'Pourquoi.'
        },
        draw(h) { /* builds the visual for this step, see below */ }
      }
    ]
  });
})();
```

Remember JS string escaping: TeX backslashes must be doubled (`'$\\sigma(a)$'`), or use template literals with `String.raw`.

## draw(h) — the rules

- `draw` is called **every time the step is shown** (arriving, going back, "Rejouer", theme change). The stage and controls are empty when it is called. It must rebuild the full picture **from scratch** for that step: steps must be **self-contained** (going directly to step 5 must look correct without having seen 1-4). Usually step k shows the state reached at end of step k-1 and animates what is new in step k.
- Animate with the toolkit only (`h.loop`, `h.tween`, `h.after`, `h.every`, `h.sleep`) — they are cancelled automatically when the user leaves the step. Never use raw `setTimeout`/`setInterval`/`requestAnimationFrame`. For async sequences use `async` + `await h.sleep(ms)` / `await h.tween(...)`; they stop silently once the step is left.
- Determinism: use `h.rng(seed)` instead of `Math.random()` so replays are identical.
- Make the visual **interactive** where meaningful (sliders, buttons, clicking/dragging points in the SVG).

## Toolkit `h`

| member | description |
|---|---|
| `h.stage`, `h.controls` | the visual container div and the controls bar div (bar hidden when empty) |
| `h.shared` | object shared by all steps of the lesson (filled in `init`) |
| `h.svg(w, hgt, parent?)` | responsive SVG with viewBox `0 0 w hgt` (use ~ 720×420), class `viz`, already has arrow markers `url(#arrow)`, `#arrow-accent`, `#arrow-pos`, `#arrow-neg`, `#arrow-muted` |
| `h.el(tag, attrs, parent)` | create SVG element; `attrs.text` = textContent, `attrs.class` = class |
| `h.attr(el, attrs)` | update attributes |
| `h.html(tag, attrs, parent)` | create HTML element |
| `h.label(svg, x, y, w, hgt, html, cls?)` | HTML + TeX inside the SVG via foreignObject (`cls` `'center'` centers it) |
| `h.tex(str, parent?, display?)` | render a TeX string into a span |
| `h.typeset(node)` | typeset `$..$` in a node |
| `h.loop(fn(t, dt))` | per-frame callback, return `false` to stop |
| `h.tween(ms, fn(p), ease?)` → Promise | p goes 0→1 (eased) |
| `h.after(ms, fn)`, `h.every(ms, fn)`, `h.sleep(ms)` | timers |
| `h.alive()` | false once left |
| `h.ease.linear/inOut/out`, `h.lerp(a,b,p)`, `h.scale(d0,d1,r0,r1)` (has `.invert`) | maths helpers |
| `h.fmt(x, d=2)` | number with French decimal comma |
| `h.rng(seed)` | deterministic random in [0,1) |
| `h.color(name)` | resolved CSS colour (for canvas). In SVG prefer `var(--name)` directly in fill/stroke |
| `h.slider({label, min, max, step, value, decimals, format, onInput})` → `{input, set, get}` | |
| `h.button(label, onClick, cls?)` | `cls: 'primary'` for the main action |
| `h.toggle([[value,label],...], value, onChange)` | segmented control |
| `h.readout(html)` → `set(html)` | small text panel in the controls bar (TeX ok) |
| `h.onLeave(fn)` | cleanup (e.g. remove a window listener) |

## Visual language (stay consistent)

- Colours via CSS variables only: `var(--ink)` text/lines, `var(--muted)` secondary/grid, `var(--line)` light borders, `var(--surface)` node fill, `var(--surface-2)` panels, `var(--accent)` current focus / signal flowing / highlighted element, `var(--accent-2)` secondary highlight (gradients, errors flowing back in backprop), `var(--pos)` positive weight / class +1 / class A, `var(--neg)` negative weight / class −1 / class B, `var(--ok)` correct, `var(--bad)` wrong, `var(--warn)` warning.
- Never hard-code hex colours (dark mode must work). White text on accent is fine.
- Neurons: circles r≈20-24, fill `var(--surface)`, stroke `var(--ink)` 1.5px. Edge thickness ∝ |weight|, colour pos/neg by sign.
- Numbers: class `mono` on SVG text; use `h.fmt`.
- Text size in SVG: 12-15 (viewBox units for a ~720 wide SVG). Keep generous margins; nothing must overflow the viewBox.
- SVG supports CSS transitions on attributes like `opacity`, `transform` via style: you can set `style="transition: opacity .4s"` then change opacity after `h.after(30, ...)`.

## Code style

- JSDoc-style `/** ... */` English summaries on functions/helpers only; **no inline `//` comments**.
- Keep each lesson file self-contained (helper functions inside the IIFE).
- Run `node --check site/lessons/<id>.js` before finishing.
