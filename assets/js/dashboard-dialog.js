/* Keep the inline catalogue usable when JavaScript or native dialogs are unavailable. */
(function () {
  'use strict';
  var catalogue = document.querySelector('.dashboard-catalogue');
  if (!catalogue || typeof document.createElement('dialog').showModal !== 'function') return;

  var dialogs = {};
  var current = null;
  var returnFocus = null;

  function triggerFor(slug) {
    return catalogue.querySelector('[data-dashboard-open="' + slug + '"]');
  }

  function fromHash() {
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (error) { return null; }
    var panel = document.getElementById(id);
    return panel && panel.getAttribute('data-dashboard-panel');
  }

  catalogue.querySelectorAll('[data-dashboard-panel]').forEach(function (panel) {
    var slug = panel.getAttribute('data-dashboard-panel');
    var title = panel.querySelector('h3');
    var dialog = document.createElement('dialog');
    dialog.className = 'dashboard-dialog';
    dialog.setAttribute('aria-labelledby', title.id);
    panel.removeAttribute('aria-labelledby');
    var bar = document.createElement('div');
    bar.className = 'dashboard-dialog-bar';
    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'dashboard-dialog-close';
    close.textContent = 'Close';
    close.setAttribute('aria-label', 'Close ' + title.textContent);
    close.addEventListener('click', function () { dialog.close(); });
    bar.appendChild(close);
    dialog.appendChild(bar);
    dialog.appendChild(panel);
    document.body.appendChild(dialog);
    dialogs[slug] = dialog;
    var navigationTarget = null;

    dialog.addEventListener('keydown', function (event) {
      if (event.key !== 'Tab') return;
      var links = dialog.querySelectorAll('button, a[href]');
      var first = links[0];
      var last = links[links.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === title)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });

    var pressedOutside = false;
    function outside(event) {
      var box = dialog.getBoundingClientRect();
      return event.target === dialog && (event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom);
    }
    dialog.addEventListener('pointerdown', function (event) { pressedOutside = outside(event); });
    dialog.addEventListener('click', function (event) {
      if (pressedOutside && outside(event)) dialog.close();
      pressedOutside = false;
    });
    dialog.addEventListener('close', function () {
      if (current !== dialog || dialog.open) return;
      current = null;
      document.documentElement.classList.remove('dashboard-dialog-open');
      if (fromHash() === slug) {
        var entry = history.state && history.state.__viDashboardDialog;
        history.replaceState(entry ? entry.previousState : history.state, '',
          entry ? entry.returnUrl : location.pathname + location.search);
      }
      var target = navigationTarget || returnFocus || triggerFor(slug);
      returnFocus = null;
      if (navigationTarget) {
        target.classList.add('dashboard-guidance-target');
        if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
        target.scrollIntoView({ block: 'start' });
        navigationTarget = null;
      } else if (target) target.focus();
    });
    // Same-page links to the retained full guidance must leave the modal first.
    dialog.addEventListener('click', function (event) {
      var link = event.target.closest('a');
      if (!link || event.defaultPrevented || event.button !== 0 ||
          event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var url = new URL(link.href);
      if (url.origin === location.origin && url.pathname === location.pathname &&
          url.search === location.search) {
        navigationTarget = document.getElementById(decodeURIComponent(url.hash.slice(1)));
        dialog.close();
      }
    });
  });
  catalogue.querySelector('[data-dashboard-details]').remove();

  function open(slug, trigger) {
    var dialog = dialogs[slug];
    if (!dialog) return;
    if (current && current !== dialog) {
      var previous = current;
      current = null;
      previous.close();
    }
    returnFocus = trigger || triggerFor(slug);
    current = dialog;
    if (!dialog.open) dialog.showModal();
    document.documentElement.classList.add('dashboard-dialog-open');
    dialog.scrollTop = 0;
    dialog.querySelector('h3').focus({ preventScroll: true });
  }

  catalogue.addEventListener('click', function (event) {
    var trigger = event.target.closest('[data-dashboard-open]');
    if (!trigger || event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var slug = trigger.getAttribute('data-dashboard-open');
    if (!dialogs[slug]) return;
    event.preventDefault();
    var previousState = history.state;
    var state = Object.assign({}, previousState);
    state.__viDashboardDialog = {
      previousState: previousState,
      returnUrl: location.pathname + location.search + location.hash
    };
    history.pushState(state, '', trigger.getAttribute('href'));
    open(slug, trigger);
  });

  function sync() {
    var slug = fromHash();
    if (slug && dialogs[slug]) open(slug);
    else if (current) current.close();
  }
  window.addEventListener('popstate', sync);
  window.addEventListener('hashchange', sync);
  if (document.readyState === 'complete') sync();
  else window.addEventListener('load', sync, { once: true });
})();
