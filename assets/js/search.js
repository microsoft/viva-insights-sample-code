/*
 * Client-side search for the Viva Insights Sample Code Library.
 * Loads a Jekyll-generated index (search.json) and queries it with Lunr.
 */
(function () {
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var status = document.getElementById("search-status");
  if (!input || !results || !status || typeof lunr === "undefined") {
    return;
  }

  var idx = null;
  var docs = [];

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch];
    });
  }

  function clip(text, max) {
    text = (text || "").trim();
    if (text.length <= max) {
      return escapeHtml(text);
    }
    return escapeHtml(text.slice(0, max).replace(/\s+\S*$/, "")) + "…";
  }

  function runQuery(raw) {
    var terms = raw
      .toLowerCase()
      .split(/\s+/)
      .map(function (t) {
        return t.replace(/[^a-z0-9]/g, "");
      })
      .filter(Boolean);

    if (!terms.length) {
      return [];
    }

    // Build a Lunr query string so the search pipeline (stemming + stopword
    // removal) is applied. Each term contributes an exact (boosted) clause,
    // a prefix clause (3+ chars), and a fuzzy clause (4+ chars) for typos.
    var queryString = terms
      .map(function (term) {
        var parts = [term + "^3"];
        if (term.length >= 3) {
          parts.push(term + "*");
        }
        if (term.length > 3) {
          parts.push(term + "~1");
        }
        return parts.join(" ");
      })
      .join(" ");

    try {
      return idx.search(queryString);
    } catch (err) {
      try {
        return idx.search(raw);
      } catch (err2) {
        return [];
      }
    }
  }

  function render() {
    var raw = input.value.trim();
    results.innerHTML = "";

    if (!raw) {
      status.textContent = "";
      return;
    }
    if (!idx) {
      status.textContent = "Loading search index…";
      return;
    }

    var matches = [];
    try {
      matches = runQuery(raw);
    } catch (err) {
      matches = [];
    }

    if (!matches.length) {
      status.textContent = 'No results for "' + raw + '".';
      return;
    }

    status.textContent =
      matches.length + (matches.length === 1 ? " result" : " results") + ' for "' + raw + '".';

    var frag = document.createDocumentFragment();
    matches.slice(0, 30).forEach(function (match) {
      var doc = docs[match.ref];
      if (!doc) {
        return;
      }
      var li = document.createElement("li");
      li.className = "vi-search-result";
      var blurb = doc.description || doc.content || "";
      li.innerHTML =
        '<a class="vi-search-link" href="' +
        encodeURI(doc.url) +
        '">' +
        escapeHtml(doc.title) +
        "</a>" +
        '<p class="vi-search-snippet">' +
        clip(blurb, 180) +
        "</p>";
      frag.appendChild(li);
    });
    results.appendChild(frag);
  }

  status.textContent = "Loading search index…";

  fetch(window.SEARCH_INDEX_URL)
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Failed to load index");
      }
      return response.json();
    })
    .then(function (data) {
      docs = data;
      idx = lunr(function () {
        this.ref("id");
        this.field("title", { boost: 10 });
        this.field("description", { boost: 5 });
        this.field("content");
        data.forEach(function (doc, i) {
          doc.id = i;
          this.add(doc);
        }, this);
      });
      status.textContent = "";
      render();
      input.focus();
    })
    .catch(function () {
      status.textContent = "Sorry — the search index could not be loaded.";
    });

  input.addEventListener("input", render);

  var initial = new URLSearchParams(window.location.search).get("q");
  if (initial) {
    input.value = initial;
  }
})();
