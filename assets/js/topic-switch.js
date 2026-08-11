/* ------------------------------------------------------------------
   Network page topic switcher: Group-to-Group / Person-to-Person / Both.
   Wires up the buttons rendered by the `.topic-switch` markup (see
   assets/css/topic-switch.css). Mirrors lang-switch.js's pattern exactly,
   but with its own storage key and data attribute so the two switches
   operate independently and can be combined on the same page.
   ------------------------------------------------------------------ */
(function () {
  var STORAGE_KEY = 'vi-network-topic-pref';

  function setTopic(topic, persist) {
    document.documentElement.setAttribute('data-topic', topic);
    document.querySelectorAll('[data-topic-btn]').forEach(function (btn) {
      var active = btn.getAttribute('data-topic-btn') === topic;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (persist) {
      try { window.localStorage.setItem(STORAGE_KEY, topic); } catch (e) {
        // localStorage can be unavailable (private browsing, disabled
        // storage). The switch still works for the current page view.
      }
    }
  }

  function init() {
    var buttons = document.querySelectorAll('[data-topic-btn]');
    if (!buttons.length) return;

    var current = document.documentElement.getAttribute('data-topic') || 'all';
    setTopic(current, false);

    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setTopic(btn.getAttribute('data-topic-btn'), true);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
