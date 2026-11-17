const modelData = {
  svm: {
    label: "SVM / REPEATED CV",
    rank: "01",
    title: "The most stable overall pipeline.",
    copy: "The SVM balances accuracy, recall, and consistency across unseen material groups.",
    accuracy: 93.60,
    f1: 93.40,
    recall: 92.82
  },
  xgboost: {
    label: "XGBOOST / REPEATED CV",
    rank: "02",
    title: "A close tree-based alternative.",
    copy: "XGBoost slightly improves mean coal recall, while remaining just behind SVM on overall stability.",
    accuracy: 93.25,
    f1: 93.14,
    recall: 93.33
  },
  pls: {
    label: "PLS-DA / REPEATED CV",
    rank: "03",
    title: "High recall, lower stability.",
    copy: "PLS-DA finds nearly all coal in grouped validation, but its performance varies strongly between materials.",
    accuracy: 82.11,
    f1: 85.37,
    recall: 99.04
  },
  forest: {
    label: "RANDOM FOREST / REPEATED CV",
    rank: "04",
    title: "A useful tree-based benchmark.",
    copy: "The forest is competitive on some groups, but trails the SVM and XGBoost in repeated validation.",
    accuracy: 92.84,
    f1: 92.36,
    recall: 91.90
  },
  cnn: {
    label: "1D CNN / FINAL HOLDOUT",
    rank: "05",
    title: "The current CNN did not generalize.",
    copy: "The raw-spectrum CNN baseline performed poorly on the unseen final materials and is not the selected model.",
    accuracy: 62.49,
    f1: 16.30,
    recall: 10.29
  }
};

const tabs = document.querySelectorAll(".model-tab");
const detailLabel = document.querySelector("#detail-label");
const detailRank = document.querySelector("#detail-rank");
const detailTitle = document.querySelector("#detail-title");
const detailCopy = document.querySelector("#detail-copy");
const metricAccuracy = document.querySelector("#metric-accuracy");
const metricF1 = document.querySelector("#metric-f1");
const metricRecall = document.querySelector("#metric-recall");
const barAccuracy = document.querySelector("#bar-accuracy");
const barF1 = document.querySelector("#bar-f1");
const barRecall = document.querySelector("#bar-recall");

function updateModel(modelKey) {
  const model = modelData[modelKey];
  if (!model) return;

  detailLabel.textContent = model.label;
  detailRank.textContent = model.rank;
  detailTitle.textContent = model.title;
  detailCopy.textContent = model.copy;
  metricAccuracy.textContent = `${model.accuracy.toFixed(2)}%`;
  metricF1.textContent = `${model.f1.toFixed(2)}%`;
  metricRecall.textContent = `${model.recall.toFixed(2)}%`;
  barAccuracy.style.width = `${model.accuracy}%`;
  barF1.style.width = `${model.f1}%`;
  barRecall.style.width = `${model.recall}%`;

  tabs.forEach((tab) => {
    const active = tab.dataset.model === modelKey;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => updateModel(tab.dataset.model));
});

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));
