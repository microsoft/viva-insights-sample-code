/* ------------------------------------------------------------------
   Global R / Python language switcher.
   Wires up the buttons rendered by the `.lang-switch` markup (see
   assets/css/lang-switch.css). A separate, tiny inline script placed right
   next to that markup in the page runs first and applies the stored
   preference (or the "r" default) to <html data-lang="..."> before this file
   loads, so the page never flashes both languages before JS is ready.
   ------------------------------------------------------------------ */
(function () {
  var STORAGE_KEY = 'vi-lang-pref';

  function setLang(lang, persist) {
    document.documentElement.setAttribute('data-lang', lang);
    document.querySelectorAll('[data-lang-btn]').forEach(function (btn) {
      var active = btn.getAttribute('data-lang-btn') === lang;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (persist) {
      try { window.localStorage.setItem(STORAGE_KEY, lang); } catch (e) {
        // localStorage can be unavailable (private browsing, disabled
        // storage). The switch still works for the current page view.
      }
    }
  }

  function init() {
    var buttons = document.querySelectorAll('[data-lang-btn]');
    if (!buttons.length) return;

    var current = document.documentElement.getAttribute('data-lang') || 'r';
    setLang(current, false);

    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setLang(btn.getAttribute('data-lang-btn'), true);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
