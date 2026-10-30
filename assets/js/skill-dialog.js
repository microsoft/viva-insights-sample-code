/* Agent Skills catalogue: progressive enhancement for _includes/skill-catalogue.html.
 *
 * Without JavaScript, each card links to an inline details section. With
 * native <dialog> support, this script moves each section into a modal dialog
 * appended to <body> and removes the inline wrapper. It runs while the page is
 * parsed, before DOMContentLoaded, so the floating TOC never sees the moved
 * headings. Hashes (#skill-<slug> and its section anchors) stay shareable.
 */
(function () {
  'use strict';

  var catalogue = document.querySelector('.skill-catalogue');
  if (!catalogue) return;

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');

  function scrollBehaviour() {
    return reduceMotion && reduceMotion.matches ? 'auto' : 'smooth';
  }

  // ---- Copy buttons (work with or without dialogs) ----
  function announce(button, message) {
    var region = button.closest('.skill-details').querySelector('.skill-copy-status');
    if (region) {
      region.textContent = '';
      window.setTimeout(function () { region.textContent = message; }, 50);
    }
  }

  function selectText(node) {
    var selection = window.getSelection();
    var range = document.createRange();
    range.selectNodeContents(node);
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function showCopied(button, label) {
    var original = button.getAttribute('data-label');
    button.textContent = label;
    button.classList.add('is-copied');
    window.clearTimeout(button._skillCopyTimer);
    button._skillCopyTimer = window.setTimeout(function () {
      button.textContent = original;
      button.classList.remove('is-copied');
    }, 2000);
  }

  function copyFrom(button) {
    var source = document.getElementById(button.getAttribute('data-copy-target'));
    if (!source) return;
    var text = source.textContent.replace(/\s+$/, '');
    var fallback = function () {
      selectText(source);
      showCopied(button, 'Press Ctrl+C or ⌘C');
      announce(button, 'Copy was blocked. The text is selected, so press Ctrl+C or Command+C to copy it.');
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () {
        showCopied(button, 'Copied');
        announce(button, 'Copied to clipboard.');
      }, fallback);
    } else {
      fallback();
    }
  }

  Array.prototype.forEach.call(catalogue.querySelectorAll('.skill-details'), function (panel) {
    var status = document.createElement('p');
    status.className = 'skill-copy-status vi-sr-only';
    status.setAttribute('aria-live', 'polite');
    panel.appendChild(status);
  });

  Array.prototype.forEach.call(catalogue.querySelectorAll('.skill-copy'), function (button) {
    button.setAttribute('data-label', button.textContent);
    button.hidden = false;
    button.addEventListener('click', function () { copyFrom(button); });
  });

  // ---- Dialog enhancement ----
  var probe = document.createElement('dialog');
  if (typeof probe.showModal !== 'function') return;

  var wrapper = catalogue.querySelector('[data-skill-details]');
  var dialogs = {};
  var current = null;
  var returnFocus = null;

  Array.prototype.forEach.call(catalogue.querySelectorAll('[data-skill-panel]'), function (panel) {
    var slug = panel.getAttribute('data-skill-panel');
    var title = panel.querySelector('.skill-details-title');
    var dialog = document.createElement('dialog');
    dialog.className = 'skill-dialog';
    dialog.setAttribute('aria-labelledby', title.id);
    // The dialog carries the name, so drop the inner region landmark.
    panel.removeAttribute('aria-labelledby');

    var bar = document.createElement('div');
    bar.className = 'skill-dialog-bar';
    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'skill-dialog-close';
    close.setAttribute('aria-label', 'Close ' + title.textContent.trim());
    close.innerHTML = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">' +
      '<path d="M3.5 3.5l9 9m0-9l-9 9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
    close.addEventListener('click', function () { dialog.close(); });
    bar.appendChild(close);

    dialog.appendChild(bar);
    dialog.appendChild(panel);
    document.body.appendChild(dialog);

    var downOutside = false;
    var outsideBox = function (event) {
      var box = dialog.getBoundingClientRect();
      return event.clientX < box.left || event.clientX > box.right ||
        event.clientY < box.top || event.clientY > box.bottom;
    };
    // Close only when both press and release land on the backdrop, so a text
    // selection dragged past the edge does not close the dialog.
    dialog.addEventListener('pointerdown', function (event) {
      downOutside = event.target === dialog && outsideBox(event);
    });
    dialog.addEventListener('click', function (event) {
      var shouldClose = downOutside && event.target === dialog && outsideBox(event);
      downOutside = false;
      if (shouldClose) dialog.close();
    });

    dialog.addEventListener('close', function () { onClose(dialog, slug); });
    dialogs[slug] = dialog;
  });

  if (wrapper) wrapper.parentNode.removeChild(wrapper);
  catalogue.classList.add('is-enhanced');

  function cardTrigger(slug) {
    return catalogue.querySelector('.skill-start[data-skill-open="' + slug + '"]');
  }

  function hashFor(slug) {
    return '#skill-' + slug;
  }

  function locate(hash) {
    if (!hash || hash.length < 2) return null;
    var id;
    try { id = decodeURIComponent(hash.slice(1)); } catch (e) { return null; }
    var target = document.getElementById(id);
    if (!target) return null;
    var dialog = target.closest('.skill-dialog');
    if (!dialog) return null;
    var panel = dialog.querySelector('[data-skill-panel]');
    return { dialog: dialog, slug: panel.getAttribute('data-skill-panel'), target: target === panel ? null : target };
  }

  function focusSection(target) {
    var heading = target.matches('h1, h2, h3, h4, h5, h6') ? target : target.querySelector('h4, h5');
    var focusTarget = heading || target;
    if (!focusTarget.hasAttribute('tabindex')) focusTarget.setAttribute('tabindex', '-1');
    target.scrollIntoView({ behavior: scrollBehaviour(), block: 'start' });
    focusTarget.focus({ preventScroll: true });
  }

  function open(slug, options) {
    var dialog = dialogs[slug];
    if (!dialog) return;
    options = options || {};

    if (current && current !== dialog) {
      var previous = current;
      current = null;
      previous.close();
    }
    if (!returnFocus || !options.keepReturn) {
      returnFocus = options.trigger || cardTrigger(slug);
    }
    if (!dialog.open) {
      dialog.showModal();
    }
    current = dialog;
    document.documentElement.classList.add('skill-dialog-open');

    if (options.target) {
      focusSection(options.target);
    } else {
      dialog.scrollTop = 0;
      dialog.querySelector('.skill-details-title').focus({ preventScroll: true });
    }
  }

  function onClose(dialog, slug) {
    if (current !== dialog || dialog.open) return;
    current = null;
    document.documentElement.classList.remove('skill-dialog-open');

    var found = locate(window.location.hash);
    if (found && found.dialog === dialog) {
      var state = history.state;
      var entry = state && state.__viSkillDialog;
      if (entry && typeof entry.returnUrl === 'string') {
        history.replaceState(entry.previousState, '', entry.returnUrl);
      } else {
        history.replaceState(state, '', window.location.pathname + window.location.search);
      }
    }

    var target = returnFocus && document.body.contains(returnFocus) ? returnFocus : cardTrigger(slug);
    returnFocus = null;
    if (target) target.focus();
  }

  function syncWithHash() {
    var found = locate(window.location.hash);
    if (found) {
      open(found.slug, { target: found.target, keepReturn: !!current });
    } else if (current) {
      current.close();
    }
  }

  document.addEventListener('click', function (event) {
    if (event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

    var jump = event.target.closest('[data-skill-jump]');
    if (jump && current && current.contains(jump)) {
      var section = document.getElementById(jump.getAttribute('href').slice(1));
      if (section) {
        event.preventDefault();
        if (window.location.hash !== jump.getAttribute('href')) {
          history.pushState(history.state, '', jump.getAttribute('href'));
        }
        focusSection(section);
      }
      return;
    }

    var opener = event.target.closest('[data-skill-open]');
    if (!opener) return;
    var slug = opener.getAttribute('data-skill-open');
    if (!dialogs[slug]) return;
    event.preventDefault();

    var fromDialog = !!(current && current.contains(opener));
    var previousUrl = window.location.pathname + window.location.search + window.location.hash;
    open(slug, { trigger: fromDialog ? null : opener, keepReturn: fromDialog });
    if (fromDialog) {
      history.replaceState(history.state, '', hashFor(slug));
    } else if (window.location.hash !== hashFor(slug)) {
      var previousState = history.state;
      var state = Object.assign({}, previousState);
      state.__viSkillDialog = { returnUrl: previousUrl, previousState: previousState };
      history.pushState(state, '', hashFor(slug));
    }
  });

  window.addEventListener('popstate', syncWithHash);
  window.addEventListener('hashchange', syncWithHash);
  // Wait for load so the browser's own fragment handling cannot reset focus.
  if (document.readyState === 'complete') {
    syncWithHash();
  } else {
    window.addEventListener('load', syncWithHash, { once: true });
  }
})();
