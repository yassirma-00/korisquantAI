# KorisQuant AI — UI/UX redesign

Visual-only redesign of the dashboard. **No backend, API, business logic,
calculation, RL algorithm, data-processing or recommendation logic was
changed.** Routes, endpoints, payloads and page behaviour are byte-identical
in intent; only presentation moved.

---

## 1 · What changed

### Design tokens (`frontend/assets/css/theme.css`)
The whole system is driven by tokens, declared twice (dark + light) so a theme
switch can never leave an orphan colour.

| Group | Before | Now |
|---|---|---|
| Surfaces | 4 steps | 5-step ladder (`--bg-0…--bg-4`) + `--bg-inset`, `--edge-light` |
| Borders | 3 | + `--border-strong` |
| Text | 3 | + `--text-3` (captions) and `--accent-text` (AA-safe accent on panels) |
| Semantics | fill only | + `--green-line` / `--red-line` / `--amber-line` outline tokens |
| Type | ad-hoc px | named scale `--fs-display … --fs-micro` |
| Spacing | ad-hoc px | 4px scale `--sp-1 … --sp-7` |
| Motion | 2 easings | + `--ease-out`, `--t-slow` |

Base palette moved to a deeper navy-black (`#05070f`) with an indigo→cyan
accent ramp, which reads as quantitative-finance tooling and keeps clear
separation from the semantic red/green used for P&L.

### Components restyled
Sidebar and navigation, header, buttons, cards, tables, dropdowns and
selectors, inputs, tabs, modals, badges, charts, AI Recommendation, Training
Monitor, Portfolio, Risk and Market pages.

Highlights:

* **Buttons** — one family, five intents (`primary`, `green`, `success-style`,
  `amber`, `red`, plus `secondary`/`ghost`), all sharing geometry, motion,
  focus ring and disabled treatment.
* **Selects** — the native OS arrow is replaced with a token-coloured chevron,
  so dropdowns finally match every other control in both themes.
* **Cards** — lit top edge (`--edge-light`); on a near-black background a
  shadow alone does not read as elevation.
* **Sidebar** — icons sit in their own tile that fills with the accent
  gradient when active; a rail marker replaces the heavy filled block.
* **AI Recommendation** — the hero now paints a semantic wash and a leading
  rail keyed to the call (`.reco-buy` / `.reco-sell` / `.reco-hold`), so
  BUY/HOLD/SELL is recognisable before a word is read. Hierarchy is explicit:
  action → confidence ring → evidence badges → price (quiet, right-aligned).

### Motion
Button hover/press/ripple, smooth dropdown and panel entries, card hover,
tab and sidebar transitions, and a page-level fade-up. All decorative;
`prefers-reduced-motion` now collapses **every** animation and transition
globally, not just a hand-listed subset.

---

## 2 · Bugs found and fixed

Both were pre-existing and are visual-layer faults, so they were in scope.

### 75 · A state-disabled button claimed to be working
`observeBusyState()` in `animate.js` toggled `.is-busy` on the bare `disabled`
attribute. Buttons disabled to express a **state** — Save on a built-in
hyperparameter profile, Compare before two checkpoints are ticked — therefore
drew a progress bar that animated **forever** on a control that was doing
nothing. The UI claimed to be working while idle.

Fixed at the root: busy now requires the disable to follow a press of that
same button (the observer runs as a microtask after the click handler, so a
genuine in-flight disable is always inside the window and a state-driven one
never is).

Verified in a real browser: `hpSaveBtn` and `tmCompareBtn` disabled-but-idle;
`genBtn` busy in flight and clean once settled.

### 76 · Disabled and loading states were illegible in light mode
Fading a gradient button with `opacity` alone measured **2.76:1** for the Save
label on the light topbar, with the button shape at **1.03:1** against its
background — the control disappeared rather than reading as unavailable. The
busy bar was a hard-coded white, invisible on the light theme's near-white
disabled surface.

Disabled now collapses every variant to one inert, tokenised surface, and the
busy bar uses `--accent`.

| | before | after |
|---|---|---|
| disabled label (dark) | 4.05:1 | **8.93:1** |
| disabled label (light) | 2.76:1 ❌ | **8.55:1** |
| button vs background | 1.03:1 | **5.01 / 5.64:1** |

### 77 · Wide tables dragged the whole page sideways on a phone
Data tables have a natural minimum width (headers are `nowrap`, figures must
not wrap mid-number). Below it the table pushed the **document**, so the page
slid under the fixed chrome and charts, disclaimer and footer drifted
off-screen — the page looked broken rather than "this table is wide".

Proven pre-existing: rebuilding the original table rules in isolation
overflows identically (657px at a 375px viewport).

Tables now scroll inside their own container, with pure-CSS edge shadows that
appear only when that direction can actually scroll.

| page @375px | before | after |
|---|---|---|
| portfolio | 425px overflow | **0** |
| rl | 325px | **0** |
| risk | 220px | **0** |
| forecast | 178px | **0** |
| signals | 48px | **0** |
| training | 6px | **0** |

### 78 · The time-range tooltip pushed the page off-screen
`.tr-info` sat in normal flow with a fixed 232px bubble, so a trigger at
x=192 on a 375px screen reached 424px and dragged the document with it.
Anchoring to either edge only moved the clipping (75–117px off the left), and
`position: fixed` does not work here — the page-transition on `.content`
leaves a `transform`, which makes a fixed child resolve against `.content`
instead of the viewport. The bubble is now anchored to the field box, which is
already inside the card and therefore inside the viewport.

### Landing / auth screens brought into the system
`landing.css` and `auth.css` were never migrated, so the **only modals and
tabs a user actually sees** still used the pre-redesign look. The auth modal
now carries the same lit top edge as dashboard cards, an inset segmented tab
strip matching the time-range control, and inputs with the system's radius,
hover and focus ring.

The generic `.modal` scaffold added in the first pass was **removed**: nothing
used it, and a dead component in a design system is a trap.

---

### 79 · The brand mark said "Fi" on every dashboard page
The rename shipped half-finished: landing and sign-in showed **K** for
KorisQuant, while all ten dashboard pages still showed **Fi** — the initials of
the name carried before the rebrand — in the sidebar tile. Signing in made the
logo change letter, which reads as two different products.

The existing rename guard never caught it: it searches for the full old product
name, and the two-letter mark is not that string.

### 80 · Navigation used typographic glyphs instead of icons
The rail rendered `◈ ◉ ◭ ⬡ ✦ ◇ ▤ ⚠ ⚙ ◎` as its icon set. Those are characters
from whatever font resolves them: inconsistent weight and baseline per
platform, no inheritable stroke width, and several carry the wrong meaning —
`⚠` is a warning sign, used merely to label the Risk page.

Replaced with a matched 16px stroke set, **inlined as SVG**: the sandboxed
preview has no network, so an icon font or CDN sprite would have failed
silently. They inherit `currentColor`, so the active tile tints them with the
accent gradient, and they are `aria-hidden` since the label carries the meaning.

Applied by `scripts/upgrade_nav_icons.py` — idempotent, and it touches only the
glyph inside `.nav-icon`, never the anchor, its `href` or its label, so
`highlightNav()` and every navigation test keep working.

Measured after the change: inactive icon contrast **4.63:1** (dark) and
**5.17:1** (light), against a 3.0:1 AA threshold for graphical components.

## 3 · Verification

* **700/700 tests pass** (7 added in total, see below); `ruff` clean; `check_install.py` 7/7.
* **0 JS errors and 0 horizontal overflow across 10 pages × 2 themes** in a real
  Chromium via Playwright.
* Every CSS selector the test-suite asserts on was checked for survival before
  and after the rewrite; all 535 original selectors are accounted for (the only
  6 removed were less-specific duplicates superseded by `:not(:disabled)`
  variants, none referenced by any test).
* Dark/light token parity asserted programmatically — no token exists in one
  theme and not the other.
* **Responsive verified**: 10 pages × {375px, 720px} — all zero horizontal
  overflow. This closes the "responsive < 720px unverified" item.

### Tests added (693 → 700)
Three regression tests, each **proven by mutation** to fail against the exact
bug it describes and to pass once restored:

| test | mutation that must fail it |
|---|---|
| `test_a_state_disabled_button_does_not_claim_to_be_working` | restore the original `toggle('is-busy', btn.disabled)` |
| `test_the_busy_bar_is_visible_on_both_themes` | hard-code the bar back to white |
| `test_a_disabled_button_stays_legible` | revert to the opacity-only fade |
| `test_the_brand_mark_is_the_same_letter_everywhere` | restore "Fi" on one page |
| `test_navigation_uses_real_icons_not_typographic_glyphs` | restore one Unicode glyph |
| `test_navigation_icons_are_inlined_and_inherit_colour` | delete the `.nav-icon svg` sizing rule |
| `test_upgrading_the_icons_left_navigation_intact` | break one `href` |

The advertised count was updated in all three places the suite pins
(`frontend/auth.html`, `frontend/landing.html`, `README.md` ×2).

---

## 4 · Files touched

| File | Change |
|---|---|
| `frontend/assets/css/theme.css` | rewritten — token system |
| `frontend/assets/css/styles.css` | rewritten — design system (24 sections) |
| `frontend/assets/js/animate.js` | busy-state root-cause fix only |
| `frontend/assets/js/pages/signals.js` | hero/sizing **markup classes only** — no API call, request shape, guard or number touched |
| `frontend/auth.html`, `landing.html`, `README.md` | test count 693 → 696 |
| `backend/tests/test_access_control.py` | +3 regression tests |
| `backend/tests/test_auth_and_brand.py` | +4 regression tests |
| `frontend/*.html` (10 pages) | brand mark `Fi`→`K`, nav glyphs → inline SVG |
| `scripts/upgrade_nav_icons.py` | new, idempotent icon migration |

No HTML page structure, no route, no endpoint and no Python application code
was modified.
