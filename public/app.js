/* ============================================
   APP.JS — Prediksi Mobil Bekas
   Pure JS: Naive Bayes, Decision Tree, KNN
   ============================================ */

(function () {
  'use strict';

  // ── State ─────────────────────────────────
  let modelData = null;

  // ── DOM refs ──────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const loadingOverlay = $('#loading-overlay');
  const formEl = $('#predict-form');
  const btnPredict = $('#btn-predict');
  const resultSection = $('#result-section');
  const toastEl = $('#toast');

  // ── Helpers ───────────────────────────────
  function formatRupiah(num) {
    return 'Rp ' + Number(num).toLocaleString('id-ID');
  }

  function formatNumber(num) {
    return Number(num).toLocaleString('id-ID');
  }

  function showToast(msg, duration = 3000) {
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    setTimeout(() => toastEl.classList.remove('show'), duration);
  }

  // ══════════════════════════════════════════
  // ML ALGORITHMS (Pure JavaScript)
  // ══════════════════════════════════════════

  // ── Naive Bayes (Gaussian) ────────────────
  function predictNaiveBayes(features, nbModel) {
    const nClasses = nbModel.class_prior.length;
    const logProbs = [];

    for (let c = 0; c < nClasses; c++) {
      let logProb = Math.log(nbModel.class_prior[c]);

      for (let i = 0; i < features.length; i++) {
        const mean = nbModel.theta[c][i];
        const variance = nbModel.var[c][i];
        // Gaussian log-likelihood
        const logLikelihood = -0.5 * Math.log(2 * Math.PI * variance)
          - 0.5 * Math.pow(features[i] - mean, 2) / variance;
        logProb += logLikelihood;
      }

      logProbs.push(logProb);
    }

    // Return class with highest log probability
    const maxIdx = logProbs.indexOf(Math.max(...logProbs));
    // Compute confidence via softmax of log probabilities
    const maxLog = Math.max(...logProbs);
    const expSum = logProbs.reduce((sum, lp) => sum + Math.exp(lp - maxLog), 0);
    const confidence = (Math.exp(logProbs[maxIdx] - maxLog) / expSum) * 100;

    return { classIdx: maxIdx, confidence: confidence.toFixed(1) };
  }

  // ── Decision Tree ─────────────────────────
  function predictDecisionTree(features, treeNode) {
    if (!treeNode) {
      // Fallback jika tree null (export gagal)
      const prior = modelData.models.naive_bayes.class_prior;
      const majorityClass = prior.indexOf(Math.max(...prior));
      return { classIdx: majorityClass, confidence: (Math.max(...prior) * 100).toFixed(1) };
    }
    if (treeNode.class !== undefined) {
      // Leaf node — hitung confidence dari distribusi kelas
      let confidence = 100;
      if (treeNode.values && treeNode.values.length > 0) {
        const total = treeNode.values.reduce((s, v) => s + v, 0);
        if (total > 0) {
          confidence = (treeNode.values[treeNode.class] / total) * 100;
        }
      }
      return { classIdx: treeNode.class, confidence: confidence.toFixed(1) };
    }

    // Internal node: compare feature to threshold
    if (features[treeNode.feature] <= treeNode.threshold) {
      return predictDecisionTree(features, treeNode.left);
    } else {
      return predictDecisionTree(features, treeNode.right);
    }
  }

  // ── KNN ───────────────────────────────────
  function predictKNN(features, knnModel) {
    const k = knnModel.k;
    const trainingData = knnModel.training_data;
    const trainingLabels = knnModel.training_labels;

    // Compute Euclidean distance to all training points
    const distances = trainingData.map((point, idx) => {
      let sumSq = 0;
      for (let i = 0; i < features.length; i++) {
        sumSq += Math.pow(features[i] - point[i], 2);
      }
      return { dist: Math.sqrt(sumSq), label: trainingLabels[idx] };
    });

    // Sort by distance, take K nearest
    distances.sort((a, b) => a.dist - b.dist);
    const kNearest = distances.slice(0, k);

    // Majority vote
    const votes = {};
    kNearest.forEach(d => {
      votes[d.label] = (votes[d.label] || 0) + 1;
    });

    let maxVotes = 0;
    let predictedClass = 0;
    for (const [label, count] of Object.entries(votes)) {
      if (count > maxVotes) {
        maxVotes = count;
        predictedClass = parseInt(label);
      }
    }

    const confidence = ((maxVotes / k) * 100).toFixed(1);
    return { classIdx: predictedClass, confidence };
  }

  // ══════════════════════════════════════════
  // INIT
  // ══════════════════════════════════════════

  let datasetRows = [];

  async function init() {
    try {
      // Load model
      const res = await fetch('models/model_data.json');
      if (!res.ok) throw new Error('File model_data.json tidak ditemukan');
      modelData = await res.json();

      // Load dataset CSV
      const csvRes = await fetch('dataset_mobil.csv');
      if (csvRes.ok) {
        const csvText = await csvRes.text();
        datasetRows = parseCSV(csvText);
      }

      renderAccuracyCards();
      renderAccuracyChart();
      populateForm();

      loadingOverlay.classList.add('hidden');
      showToast('✅ Model berhasil dimuat — siap prediksi!');
    } catch (err) {
      console.error('Init error:', err);
      loadingOverlay.querySelector('.loading-text').textContent = 'Gagal memuat model';
      loadingOverlay.querySelector('.loading-sub').textContent = err.message;
      loadingOverlay.querySelector('.loading-spinner').style.borderTopColor = '#ef4444';
    }
  }

  // ── Parse CSV ─────────────────────────────
  function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(';').map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(';');
      if (vals.length < headers.length) continue;
      const row = {};
      headers.forEach((h, idx) => row[h] = vals[idx].trim());
      rows.push(row);
    }
    return rows;
  }

  // ── Compute mileage/price category bins (quartile-based) ──
  function computeCategories(rows) {
    // Compute quartile-based bins for mileage
    const mileages = rows.map(r => parseInt(r.mileage)).sort((a, b) => a - b);
    const prices = rows.map(r => parseFloat(r.price)).sort((a, b) => a - b);

    function quartileBins(sorted) {
      const n = sorted.length;
      const q1 = sorted[Math.floor(n * 0.25)];
      const q2 = sorted[Math.floor(n * 0.5)];
      const q3 = sorted[Math.floor(n * 0.75)];
      const min = sorted[0];
      const max = sorted[n - 1];
      return [
        { low: min - 0.001, high: q1, label: `(${(min - 0.001).toFixed(1)}, ${q1.toFixed(1)}]` },
        { low: q1, high: q2, label: `(${q1.toFixed(1)}, ${q2.toFixed(1)}]` },
        { low: q2, high: q3, label: `(${q2.toFixed(1)}, ${q3.toFixed(1)}]` },
        { low: q3, high: max, label: `(${q3.toFixed(1)}, ${max.toFixed(1)}]` }
      ];
    }

    const mileageBins = quartileBins(mileages);
    const priceBins = quartileBins(prices);

    // Assign categories to each row
    rows.forEach(row => {
      const m = parseInt(row.mileage);
      const p = parseFloat(row.price);

      row.mileage_category = mileageBins.find(b => m > b.low && m <= b.high)?.label || mileageBins[mileageBins.length - 1].label;
      row.price_category = priceBins.find(b => p > b.low && p <= b.high)?.label || priceBins[priceBins.length - 1].label;
    });
  }

  // ── DOM refs for cascading dropdowns ───────
  function getSelects() {
    return {
      brand: $('#input-brand'),
      trans: $('#input-transmission'),
      year: $('#input-year'),
      mileage: $('#input-mileage'),
      price: $('#input-price'),
      target: $('#input-target'),
      mileageCat: $('#input-mileage-cat'),
      priceCat: $('#input-price-cat'),
    };
  }

  // ── Utility: reset a <select> to only its placeholder ──
  function resetSelect(sel, placeholder) {
    sel.innerHTML = `<option value="">${placeholder}</option>`;
    sel.disabled = true;
  }

  // ── Utility: populate a <select> with string options ──
  function fillSelect(sel, options, formatter) {
    options.forEach(val => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = formatter ? formatter(val) : val;
      sel.appendChild(opt);
    });
    sel.disabled = false;
  }

  // ── Filter dataset rows by current selections ──
  function filterRows(filters) {
    return datasetRows.filter(row => {
      if (filters.brand && row.car_brand !== filters.brand) return false;
      if (filters.trans && row.transmission !== filters.trans) return false;
      if (filters.year && parseInt(row.year_of_manufacture) !== parseInt(filters.year)) return false;
      if (filters.mileage && parseInt(row.mileage) !== parseInt(filters.mileage)) return false;
      if (filters.price && parseInt(parseFloat(row.price)) !== parseInt(filters.price)) return false;
      return true;
    });
  }

  // ── Cascading update functions ─────────────

  function onBrandChange() {
    const s = getSelects();
    const brand = s.brand.value;

    // Reset all downstream
    resetSelect(s.trans, 'Pilih Transmisi');
    resetSelect(s.year, 'Pilih Tahun');
    resetSelect(s.mileage, 'Pilih Kilometer');
    resetSelect(s.price, 'Pilih Harga');
    resetSelect(s.target, 'Pilih Target');
    resetSelect(s.mileageCat, 'Pilih Mileage Category');
    resetSelect(s.priceCat, 'Pilih Price Category');

    if (!brand) return;

    const filtered = filterRows({ brand });
    const transmissions = [...new Set(filtered.map(r => r.transmission))].sort();
    fillSelect(s.trans, transmissions);
  }

  function onTransChange() {
    const s = getSelects();
    const brand = s.brand.value;
    const trans = s.trans.value;

    resetSelect(s.year, 'Pilih Tahun');
    resetSelect(s.mileage, 'Pilih Kilometer');
    resetSelect(s.price, 'Pilih Harga');
    resetSelect(s.target, 'Pilih Target');
    resetSelect(s.mileageCat, 'Pilih Mileage Category');
    resetSelect(s.priceCat, 'Pilih Price Category');

    if (!trans) return;

    const filtered = filterRows({ brand, trans });
    const years = [...new Set(filtered.map(r => parseInt(r.year_of_manufacture)))].sort((a, b) => b - a);
    fillSelect(s.year, years);
  }

  function onYearChange() {
    const s = getSelects();
    const brand = s.brand.value;
    const trans = s.trans.value;
    const year = s.year.value;

    resetSelect(s.mileage, 'Pilih Kilometer');
    resetSelect(s.price, 'Pilih Harga');
    resetSelect(s.target, 'Pilih Target');
    resetSelect(s.mileageCat, 'Pilih Mileage Category');
    resetSelect(s.priceCat, 'Pilih Price Category');

    if (!year) return;

    const filtered = filterRows({ brand, trans, year });
    const mileages = [...new Set(filtered.map(r => parseInt(r.mileage)))].sort((a, b) => a - b);
    fillSelect(s.mileage, mileages, m => formatNumber(m) + ' km');
  }

  function onMileageChange() {
    const s = getSelects();
    const brand = s.brand.value;
    const trans = s.trans.value;
    const year = s.year.value;
    const mileage = s.mileage.value;

    resetSelect(s.price, 'Pilih Harga');
    resetSelect(s.target, 'Pilih Target');
    resetSelect(s.mileageCat, 'Pilih Mileage Category');
    resetSelect(s.priceCat, 'Pilih Price Category');

    if (!mileage) return;

    const filtered = filterRows({ brand, trans, year, mileage });
    const prices = [...new Set(filtered.map(r => parseInt(parseFloat(r.price))))].sort((a, b) => a - b);
    fillSelect(s.price, prices, p => formatRupiah(p));
  }

  function onPriceChange() {
    const s = getSelects();
    const brand = s.brand.value;
    const trans = s.trans.value;
    const year = s.year.value;
    const mileage = s.mileage.value;
    const price = s.price.value;

    resetSelect(s.target, 'Pilih Target');
    resetSelect(s.mileageCat, 'Pilih Mileage Category');
    resetSelect(s.priceCat, 'Pilih Price Category');

    if (!price) return;

    const filtered = filterRows({ brand, trans, year, mileage, price });

    // Target
    const targets = [...new Set(filtered.map(r => r.target))].sort();
    fillSelect(s.target, targets);
    if (targets.length === 1) s.target.value = targets[0];

    // Mileage Category
    const mileageCats = [...new Set(filtered.map(r => r.mileage_category))].sort();
    fillSelect(s.mileageCat, mileageCats);
    if (mileageCats.length === 1) s.mileageCat.value = mileageCats[0];

    // Price Category
    const priceCats = [...new Set(filtered.map(r => r.price_category))].sort();
    fillSelect(s.priceCat, priceCats);
    if (priceCats.length === 1) s.priceCat.value = priceCats[0];
  }

  // ── Populate form dropdowns (cascading) ───────────
  function populateForm() {
    const s = getSelects();

    // Compute categories for all rows
    if (datasetRows.length > 0) {
      computeCategories(datasetRows);
    }

    // Brand — always populated from dataset
    if (datasetRows.length > 0) {
      const brands = [...new Set(datasetRows.map(r => r.car_brand))].sort();
      brands.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b; opt.textContent = b;
        s.brand.appendChild(opt);
      });
    } else {
      // Fallback from model metadata
      const meta = modelData.metadata;
      meta.brand_classes.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b; opt.textContent = b;
        s.brand.appendChild(opt);
      });
    }

    // All downstream start disabled
    s.trans.disabled = true;
    s.year.disabled = true;
    s.mileage.disabled = true;
    s.price.disabled = true;
    s.target.disabled = true;
    s.mileageCat.disabled = true;
    s.priceCat.disabled = true;

    // Attach cascading event listeners
    s.brand.addEventListener('change', onBrandChange);
    s.trans.addEventListener('change', onTransChange);
    s.year.addEventListener('change', onYearChange);
    s.mileage.addEventListener('change', onMileageChange);
    s.price.addEventListener('change', onPriceChange);
  }

  // ── Predict ───────────────────────────────
  function predict() {
    const brand = $('#input-brand').value;
    const transmission = $('#input-transmission').value;
    const year = parseInt($('#input-year').value);
    const mileage = parseInt($('#input-mileage').value);
    const price = parseFloat($('#input-price').value);
    const target = $('#input-target').value;
    const mileageCat = $('#input-mileage-cat').value;
    const priceCat = $('#input-price-cat').value;

    if (!brand || !transmission || !year || isNaN(mileage) || isNaN(price) || !target || !mileageCat || !priceCat) {
      showToast('⚠️ Lengkapi semua field terlebih dahulu');
      return;
    }

    btnPredict.classList.add('loading');
    btnPredict.disabled = true;

    try {
      const meta = modelData.metadata;
      const models = modelData.models;

      const brandIdx = meta.brand_classes.indexOf(brand);
      const transIdx = meta.transmission_classes.indexOf(transmission);
      if (brandIdx === -1 || transIdx === -1) {
        showToast('⚠️ Brand atau transmisi tidak valid');
        return;
      }

      const rawFeatures = [brandIdx, transIdx, year, mileage, price];
      const scaled = rawFeatures.map((val, i) =>
        (val - meta.scaler_mean[i]) / meta.scaler_scale[i]
      );

      const nbResult = predictNaiveBayes(scaled, models.naive_bayes);
      const dtResult = predictDecisionTree(scaled, models.decision_tree.tree);
      const knnResult = predictKNN(scaled, models.knn);
      const targetClasses = meta.target_classes;

      renderResult({
        nb: { prediction: targetClasses[nbResult.classIdx], confidence: nbResult.confidence },
        dt: { prediction: targetClasses[dtResult.classIdx], confidence: dtResult.confidence },
        knn: { prediction: targetClasses[knnResult.classIdx], confidence: knnResult.confidence },
        inputs: { brand, transmission, year, mileage, price, target, mileageCat, priceCat }
      });

    } catch (err) {
      console.error('Predict error:', err);
      showToast('❌ Error: ' + err.message);
    } finally {
      btnPredict.classList.remove('loading');
      btnPredict.disabled = false;
    }
  }

  // ── Render Result ─────────────────────────
  function renderResult(data) {
    function cardClass(pred) {
      return pred === 'Cepat Terjual' ? 'fast' : 'slow';
    }
    function cardEmoji(pred) {
      return pred === 'Cepat Terjual' ? '🚀' : '⏳';
    }

    resultSection.innerHTML = `
      <h2 class="section-title">
        <span class="icon">📊</span>
        Hasil Prediksi 3 Algoritma
      </h2>

      <div class="prediction-grid">
        <div class="prediction-card">
          <div class="prediction-card-header">Naive Bayes</div>
          <div class="result-icon ${cardClass(data.nb.prediction)}" style="width:56px;height:56px;font-size:1.5rem;margin:12px auto;">
            ${cardEmoji(data.nb.prediction)}
          </div>
          <div class="result-value ${cardClass(data.nb.prediction)}" style="font-size:1.2rem;">${data.nb.prediction}</div>
          <div class="result-confidence">Confidence: ${data.nb.confidence}%</div>
        </div>

        <div class="prediction-card">
          <div class="prediction-card-header">Decision Tree</div>
          <div class="result-icon ${cardClass(data.dt.prediction)}" style="width:56px;height:56px;font-size:1.5rem;margin:12px auto;">
            ${cardEmoji(data.dt.prediction)}
          </div>
          <div class="result-value ${cardClass(data.dt.prediction)}" style="font-size:1.2rem;">${data.dt.prediction}</div>
          <div class="result-confidence">Confidence: ${data.dt.confidence}%</div>
        </div>

        <div class="prediction-card">
          <div class="prediction-card-header">KNN</div>
          <div class="result-icon ${cardClass(data.knn.prediction)}" style="width:56px;height:56px;font-size:1.5rem;margin:12px auto;">
            ${cardEmoji(data.knn.prediction)}
          </div>
          <div class="result-value ${cardClass(data.knn.prediction)}" style="font-size:1.2rem;">${data.knn.prediction}</div>
          <div class="result-confidence">Confidence: ${data.knn.confidence}%</div>
        </div>
      </div>

      <h3 style="font-size:0.88rem; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.6px; margin-bottom:14px; margin-top:24px;">
        Data Input
      </h3>
      <div class="summary-grid">
        <div class="summary-item">
          <div class="label">Merek</div>
          <div class="value">${data.inputs.brand}</div>
        </div>
        <div class="summary-item">
          <div class="label">Transmisi</div>
          <div class="value">${data.inputs.transmission}</div>
        </div>
        <div class="summary-item">
          <div class="label">Tahun</div>
          <div class="value">${data.inputs.year}</div>
        </div>
        <div class="summary-item">
          <div class="label">Kilometer</div>
          <div class="value">${formatNumber(data.inputs.mileage)} km</div>
        </div>
        <div class="summary-item">
          <div class="label">Harga</div>
          <div class="value">${formatRupiah(data.inputs.price)}</div>
        </div>
        <div class="summary-item">
          <div class="label">Target</div>
          <div class="value">${data.inputs.target}</div>
        </div>
        <div class="summary-item">
          <div class="label">Mileage Category</div>
          <div class="value">${data.inputs.mileageCat}</div>
        </div>
        <div class="summary-item">
          <div class="label">Price Category</div>
          <div class="value">${data.inputs.priceCat}</div>
        </div>
      </div>
    `;

    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // ── Render Accuracy Cards ─────────────────
  function renderAccuracyCards() {
    const grid = $('#accuracy-grid');
    const accs = modelData.accuracies;
    const models = [
      { name: 'Naive Bayes', key: 'naive_bayes' },
      { name: 'Decision Tree', key: 'decision_tree' },
      { name: 'KNN', key: 'knn' }
    ];

    grid.innerHTML = models.map(m => {
      const a = accs[m.key];
      const accPct = (a.accuracy * 100).toFixed(1);
      return `
        <div class="accuracy-card">
          <div class="model-name">${m.name}</div>
          <div class="accuracy-value">${accPct}%</div>
          <div class="accuracy-bar">
            <div class="fill" data-width="${accPct}"></div>
          </div>
          <div style="margin-top:12px; font-size:0.75rem; color:var(--text-muted);">
            P: ${(a.precision*100).toFixed(1)}% &nbsp;|&nbsp;
            R: ${(a.recall*100).toFixed(1)}% &nbsp;|&nbsp;
            F1: ${(a.f1_score*100).toFixed(1)}%
          </div>
        </div>
      `;
    }).join('');

    setTimeout(() => {
      grid.querySelectorAll('.fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 400);
  }

  // ── Render Accuracy Chart ─────────────────
  function renderAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    const accs = modelData.accuracies;

    const labels = ['Naive Bayes', 'Decision Tree', 'KNN'];
    const keys = ['naive_bayes', 'decision_tree', 'knn'];
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const metricLabels = ['Akurasi', 'Precision', 'Recall', 'F1-Score'];
    const colors = ['#d97706', '#10b981', '#059669', '#ea580c'];

    const datasets = metrics.map((metric, i) => ({
      label: metricLabels[i],
      data: keys.map(k => (accs[k][metric] * 100).toFixed(1)),
      backgroundColor: colors[i] + '44',
      borderColor: colors[i],
      borderWidth: 2,
      borderRadius: 6,
      borderSkipped: false
    }));

    new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        animation: { duration: 1200, easing: 'easeOutQuart' },
        plugins: {
          legend: {
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
          },
          tooltip: {
            backgroundColor: 'rgba(17, 24, 39, 0.95)',
            titleColor: '#f1f5f9',
            bodyColor: '#94a3b8',
            borderColor: 'rgba(217,119,6,0.2)',
            borderWidth: 1,
            cornerRadius: 10,
            padding: 12,
            callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%` }
          }
        },
        scales: {
          y: {
            beginAtZero: true, max: 100,
            grid: { color: 'rgba(217,119,6,0.06)', drawBorder: false },
            ticks: { color: '#64748b', font: { size: 11, family: 'Inter' }, callback: v => v + '%' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', font: { size: 11, weight: 600, family: 'Inter' } }
          }
        }
      }
    });
  }

  // ── Events ────────────────────────────────
  formEl.addEventListener('submit', (e) => {
    e.preventDefault();
    predict();
  });

  // ── Init ──────────────────────────────────
  init();
})();
