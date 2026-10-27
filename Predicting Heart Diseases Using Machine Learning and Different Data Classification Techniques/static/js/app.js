// Small page helpers shared by every template. No libraries.

// ── Dropdowns ──────────────────────────────────────────────────────────────
// Each <div data-dd> holds a native <select>. We draw a styled button + list on top of it and keep
// the two in sync. The <select> stays in the page (invisible) so the form still submits it and the
// browser's "required" check still works. Keyboard: ↑ ↓ Home End move, Enter/Space pick,
// Esc closes, typing a letter jumps to the next option starting with it.
document.querySelectorAll('[data-dd]').forEach(dd => {
   const select = dd.querySelector('select');
   const options = [...select.options].filter(o => !o.disabled);
   select.tabIndex = -1;
   select.setAttribute('aria-hidden', 'true');

   const button = document.createElement('button');
   button.type = 'button';
   button.className = 'dd-btn';
   button.setAttribute('aria-haspopup', 'listbox');
   button.setAttribute('aria-expanded', 'false');
   const label = document.querySelector(`label[for="${select.id}"]`);
   if (label) { label.htmlFor = ''; label.id = `${select.id}-label`; label.addEventListener('click', () => button.focus()); }
   button.setAttribute('aria-labelledby', `${select.id}-label ${select.id}-value`);
   button.innerHTML = `<span class="dd-value" id="${select.id}-value"></span><svg viewBox="0 0 12 8" aria-hidden="true"><path d="M1 1.5l5 5 5-5"/></svg>`;

   const list = document.createElement('ul');
   list.className = 'dd-list';
   list.setAttribute('role', 'listbox');
   list.id = `${select.id}-list`;
   list.hidden = true;
   button.setAttribute('aria-controls', list.id);
   options.forEach((o, i) => {
      const li = document.createElement('li');
      li.setAttribute('role', 'option');
      li.id = `${select.id}-opt-${i}`;
      li.dataset.value = o.value;
      li.innerHTML = '<b></b>' + (o.dataset.desc ? '<small></small>' : '');
      li.querySelector('b').textContent = o.textContent;
      if (o.dataset.desc) li.querySelector('small').textContent = o.dataset.desc;
      li.addEventListener('mousedown', e => e.preventDefault());  // keep focus on the button
      li.addEventListener('click', () => { choose(i); close(); });
      list.appendChild(li);
   });
   dd.append(button, list);
   const items = [...list.children];
   let active = -1;

   function sync() {
      const i = options.findIndex(o => o.value === select.value && select.value !== '');
      button.querySelector('.dd-value').textContent = i >= 0 ? options[i].textContent : 'Choose…';
      button.classList.toggle('placeholder', i < 0);
      items.forEach((li, j) => li.setAttribute('aria-selected', String(j === i)));
      return i;
   }
   function highlight(i) {
      active = Math.max(0, Math.min(items.length - 1, i));
      items.forEach((li, j) => li.classList.toggle('active', j === active));
      button.setAttribute('aria-activedescendant', items[active].id);
      items[active].scrollIntoView({ block: 'nearest' });
   }
   function choose(i) {
      select.value = options[i].value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
   }
   function open() {
      document.querySelectorAll('.dd-list:not([hidden])').forEach(l => l !== list && l.dispatchEvent(new Event('dd-close')));
      list.hidden = false;
      dd.classList.add('open');
      button.setAttribute('aria-expanded', 'true');
      highlight(Math.max(0, sync()));
   }
   function close() {
      list.hidden = true;
      dd.classList.remove('open');
      button.setAttribute('aria-expanded', 'false');
      button.removeAttribute('aria-activedescendant');
   }

   list.addEventListener('dd-close', close);
   select.addEventListener('change', sync);
   button.addEventListener('click', () => (list.hidden ? open() : close()));
   button.addEventListener('blur', close);
   button.addEventListener('keydown', e => {
      const isOpen = !list.hidden;
      if (['ArrowDown', 'ArrowUp'].includes(e.key)) {
         e.preventDefault();
         if (!isOpen) return open();
         highlight(active + (e.key === 'ArrowDown' ? 1 : -1));
      } else if (e.key === 'Home' || e.key === 'End') {
         if (!isOpen) return;
         e.preventDefault();
         highlight(e.key === 'Home' ? 0 : items.length - 1);
      } else if (e.key === 'Enter' || e.key === ' ') {
         e.preventDefault();
         if (!isOpen) return open();
         choose(active); close();
      } else if (e.key === 'Escape' && isOpen) {
         e.preventDefault(); close();
      } else if (e.key.length === 1 && /\S/.test(e.key)) {
         const start = isOpen ? active + 1 : sync() + 1;
         const order = [...options.keys()].map(k => (k + start) % options.length);
         const hit = order.find(k => options[k].textContent.toLowerCase().startsWith(e.key.toLowerCase()));
         if (hit === undefined) return;
         if (isOpen) highlight(hit); else choose(hit);
      }
   });
   // When the browser's required-check fails on the hidden select, move focus to the visible button.
   select.addEventListener('invalid', () => { dd.classList.add('invalid'); setTimeout(() => button.focus()); });
   select.addEventListener('change', () => dd.classList.remove('invalid'));
   select.form.addEventListener('reset', () => setTimeout(() => { sync(); dd.classList.remove('invalid'); }));
   sync();
});

// ── "Random patient" buttons ───────────────────────────────────────────────
// Fill the form with a real row from heart.csv. The button carries the field names and rows;
// each row ends with that patient's real diagnosis. form.elements[name] works for inputs,
// selects and radio groups alike.
document.querySelectorAll('[data-random]').forEach(btn => {
   const fields = JSON.parse(btn.dataset.fields), rows = JSON.parse(btn.dataset.rows);
   const note = document.getElementById(btn.dataset.note);
   btn.addEventListener('click', () => {
      const i = Math.floor(Math.random() * rows.length), row = rows[i];
      fields.forEach((f, j) => {
         const el = btn.form.elements[f];
         el.value = String(row[j]);
         if (el.dispatchEvent) el.dispatchEvent(new Event('change', { bubbles: true }));
      });
      btn.form.elements.row.value = i;
      note.textContent = `Patient ${i + 1} of ${rows.length} from heart.csv. In the dataset this patient `
         + (row[fields.length] === 1 ? 'has heart disease.' : 'does not have heart disease.');
      note.hidden = false;
   });
   btn.form.addEventListener('reset', () => { note.hidden = true; });  // reset also restores the hidden row to its default
});

// ── Info (i) icons ─────────────────────────────────────────────────────────
// Hover or keyboard focus shows the note (pure CSS). Tapping toggles it, for touch screens;
// tapping elsewhere or pressing Esc closes it.
document.addEventListener('click', e => {
   const btn = e.target.closest('.info');
   document.querySelectorAll('.info.open').forEach(b => b !== btn && b.classList.remove('open'));
   if (btn) btn.classList.toggle('open');
});
document.addEventListener('keydown', e => {
   if (e.key === 'Escape') document.querySelectorAll('.info.open').forEach(b => b.classList.remove('open'));
});

// ── Print / save as PDF, and "are you sure?" before destructive forms ─────────
document.querySelectorAll('[data-print]').forEach(b => b.addEventListener('click', () => window.print()));
document.querySelectorAll('form[data-confirm]').forEach(f => f.addEventListener('submit', e => {
   if (!confirm(f.dataset.confirm)) e.preventDefault();
}));

// ── Theme toggle ───────────────────────────────────────────────────────────
// Flips light/dark and remembers the choice in this browser.
document.getElementById('theme-toggle').addEventListener('click', () => {
   const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
   document.documentElement.dataset.theme = next;
   try { localStorage.setItem('theme', next); } catch (e) { /* private window: still switches, just isn't remembered */ }
});

// ── Chart tooltips ─────────────────────────────────────────────────────────
// For SVG chart points (HTML marks use the CSS [data-tip] tooltip instead).
const tip = document.createElement('div');
tip.className = 'svg-tip';
tip.hidden = true;
document.body.appendChild(tip);
document.querySelectorAll('svg [data-tip]').forEach(el => {
   el.addEventListener('mouseenter', () => { tip.textContent = el.dataset.tip; tip.hidden = false; });
   el.addEventListener('mousemove', e => { tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY - 36) + 'px'; });
   el.addEventListener('mouseleave', () => { tip.hidden = true; });
});
