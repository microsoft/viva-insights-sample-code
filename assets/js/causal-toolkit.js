/* ------------------------------------------------------------------
   Copilot Causal Toolkit — accessible tab widget.
   Markup:
     <div class="ct-tabs" data-ct-tabs>
       <div class="ct-tablist" role="tablist" aria-label="...">
         <button class="ct-tab" role="tab">Label</button> ...
       </div>
       <div class="ct-panel" role="tabpanel" markdown="1"> ... </div> ...
     </div>
   The Nth button controls the Nth panel. First tab/panel is active by default.
   ------------------------------------------------------------------ */
(function () {
  function initTabs(root) {
    var tablist = root.querySelector('.ct-tablist');
    if (!tablist) return;
    var tabs = Array.prototype.slice.call(tablist.querySelectorAll('.ct-tab'));
    var panels = Array.prototype.slice.call(root.querySelectorAll('.ct-panel'));
    if (!tabs.length || tabs.length !== panels.length) return;

    function select(index, focus) {
      tabs.forEach(function (tab, i) {
        var active = i === index;
        tab.classList.toggle('is-active', active);
        tab.setAttribute('aria-selected', active ? 'true' : 'false');
        tab.setAttribute('tabindex', active ? '0' : '-1');
        panels[i].hidden = !active;
        if (active && focus) tab.focus();
      });
    }

    tabs.forEach(function (tab, i) {
      tab.setAttribute('role', 'tab');
      tab.setAttribute('id', tab.id || 'ct-tab-' + Math.random().toString(36).slice(2, 8));
      panels[i].setAttribute('role', 'tabpanel');
      panels[i].setAttribute('aria-labelledby', tab.id);
      tab.addEventListener('click', function () { select(i); });
      tab.addEventListener('keydown', function (e) {
        var next;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') next = 0;
        else if (e.key === 'End') next = tabs.length - 1;
        else return;
        e.preventDefault();
        select(next, true);
      });
    });

    select(0);
  }

  function init() {
    document.querySelectorAll('[data-ct-tabs]').forEach(initTabs);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
