(function () {
  'use strict';

  var catalogue = document.querySelector('[data-prompt-catalogue]');
  if (!catalogue) return;

  var panels = catalogue.querySelectorAll('[data-prompt-panel]');
  var probe = document.createElement('dialog');

  var dialogs = {};
  var current = null;
  var returnFocus = null;

  function slugFromHash() {
    var match = window.location.hash.match(/^#prompt-([a-z0-9-]+)$/);
    return match ? match[1] : null;
  }

  function closeButton(title, dialog) {
    var bar = document.createElement('div');
    bar.className = 'prompt-dialog-bar';
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'prompt-dialog-close';
    button.setAttribute('aria-label', 'Close ' + title);
    button.innerHTML = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">' +
      '<path d="M3.5 3.5l9 9m0-9l-9 9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
    button.addEventListener('click', function () { dialog.close(); });
    bar.appendChild(button);
    return bar;
  }

  function announce(button, message) {
    var region = button.closest('.prompt-details').querySelector('.prompt-copy-status');
    if (!region) return;
    region.textContent = '';
    window.setTimeout(function () { region.textContent = message; }, 50);
  }

  function selectText(node) {
    var selection = window.getSelection();
    var range = document.createRange();
    range.selectNodeContents(node);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function copyPrompt(button) {
    var source = document.getElementById(button.getAttribute('data-copy-target'));
    if (!source) return;
    var original = button.getAttribute('data-label');
    var fallback = function () {
      selectText(source);
      button.textContent = 'Press Ctrl+C or ⌘C';
      announce(button, 'Copy was blocked. The prompt is selected, so press Ctrl+C or Command+C to copy it.');
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(source.textContent.replace(/\s+$/, '')).then(function () {
        button.textContent = 'Copied';
        button.classList.add('is-copied');
        announce(button, 'Quick prompt copied to clipboard.');
        window.setTimeout(function () {
          button.textContent = original;
          button.classList.remove('is-copied');
        }, 2000);
      }, fallback);
    } else {
      fallback();
    }
  }

  Array.prototype.forEach.call(panels, function (panel) {
    var status = document.createElement('p');
    status.className = 'prompt-copy-status vi-sr-only';
    status.setAttribute('aria-live', 'polite');
    panel.appendChild(status);

    var copy = panel.querySelector('.prompt-copy');
    copy.setAttribute('data-label', copy.textContent);
    copy.hidden = false;
    copy.addEventListener('click', function () { copyPrompt(copy); });
  });

  if (typeof probe.showModal !== 'function') return;

  Array.prototype.forEach.call(panels, function (panel) {
    var slug = panel.getAttribute('data-prompt-panel');
    var title = panel.querySelector('.prompt-details-title');
    var dialog = document.createElement('dialog');
    dialog.className = 'prompt-dialog';
    dialog.setAttribute('aria-labelledby', title.id);
    panel.removeAttribute('aria-labelledby');
    dialog.appendChild(closeButton(title.textContent.trim(), dialog));
    dialog.appendChild(panel);
    document.body.appendChild(dialog);

    var pointerStartedOutside = false;
    function outside(event) {
      var box = dialog.getBoundingClientRect();
      return event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom;
    }
    dialog.addEventListener('pointerdown', function (event) {
      pointerStartedOutside = event.target === dialog && outside(event);
    });
    dialog.addEventListener('click', function (event) {
      var close = pointerStartedOutside && event.target === dialog && outside(event);
      pointerStartedOutside = false;
      if (close) dialog.close();
    });

    dialog.addEventListener('close', function () {
      if (current !== dialog || dialog.open) return;
      current = null;
      document.documentElement.classList.remove('prompt-dialog-open');
      if (slugFromHash() === slug) {
        if (window.history.state && window.history.state.__viPromptDialog === slug) {
          window.history.back();
        } else {
          window.history.replaceState(window.history.state, '', window.location.pathname + window.location.search);
        }
      }
      var trigger = returnFocus || catalogue.querySelector('[data-prompt-open="' + slug + '"]');
      returnFocus = null;
      if (trigger) trigger.focus();
    });
    dialogs[slug] = dialog;
  });

  var fallback = catalogue.querySelector('[data-prompt-details]');
  if (fallback) fallback.remove();

  function open(slug, trigger) {
    var dialog = dialogs[slug];
    if (!dialog) return;
    if (current && current !== dialog) current.close();
    current = dialog;
    returnFocus = trigger || catalogue.querySelector('[data-prompt-open="' + slug + '"]');
    if (!dialog.open) {
      dialog.showModal();
      dialog.scrollTop = 0;
    }
    document.documentElement.classList.add('prompt-dialog-open');
    dialog.querySelector('.prompt-details-title').focus({ preventScroll: true });
  }

  function syncHash() {
    var slug = slugFromHash();
    if (slug && dialogs[slug]) {
      if (current !== dialogs[slug] || !dialogs[slug].open) open(slug);
    } else if (current) {
      current.close();
    }
  }

  catalogue.addEventListener('click', function (event) {
    if (event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    var opener = event.target.closest('[data-prompt-open]');
    if (!opener) return;
    var slug = opener.getAttribute('data-prompt-open');
    if (!dialogs[slug]) return;
    event.preventDefault();
    if (window.location.hash !== '#prompt-' + slug) {
      var state = Object.assign({}, window.history.state);
      state.__viPromptDialog = slug;
      window.history.pushState(state, '', '#prompt-' + slug);
    }
    open(slug, opener);
  });

  window.addEventListener('popstate', syncHash);
  window.addEventListener('hashchange', syncHash);
  if (document.readyState === 'complete') {
    syncHash();
  } else {
    window.addEventListener('load', syncHash, { once: true });
  }
  catalogue.classList.add('is-enhanced');
})();
