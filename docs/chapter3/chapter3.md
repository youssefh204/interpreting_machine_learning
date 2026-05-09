# Chapter 3 — Feature-based Explanations

**Author: [Youssef Hemaya](https://www.linkedin.com/in/youssef-hemaya/)**  
*CSEN 1153 — Seminar on Explaining AI | German University in Cairo | Spring 2026*

---

Imagine you apply for a loan. You fill out the form, submit your documents, and wait. Then a message arrives: **Denied.** No reason. No explanation. Just a decision made by an algorithm that has seen thousands of applications before yours.

You'd want to know why, right?

That's exactly what feature-based explanations try to answer. They look inside a model's decision — not by cracking it open like a black box — but by asking a simpler question: *which parts of the input actually mattered?*

In this chapter, we'll explore three methods that do this in three very different ways: **Permutation Importance**, **LIME**, and **SHAP**. By the end, you'll understand what each one does, when to use it, and — just as importantly — when *not* to trust it.

---

## 1. Why Do We Even Need Feature Explanations?

Before we jump into the methods, let's think about who actually needs these explanations and why.

**The person who was denied the loan** wants to know what they can do differently. Was it their income? Their credit history? Their age?

**The regulator** wants to make sure the bank isn't discriminating. In Europe, GDPR Article 22 legally requires that people receive a meaningful explanation when an automated system makes a decision about them. The EU AI Act (2024) goes further — for high-risk AI systems, transparency isn't optional.

**The data scientist** who built the model wants to debug it. A model can achieve 95% accuracy on test data and still be learning the wrong things — like using zip codes as a proxy for ethnicity, or penalizing people who call the bank too often (a sign of financial distress that the model shouldn't be exploiting).

Feature-based explanations serve all three of these audiences. They're tools for trust, accountability, and debugging — all at once.

---

## 2. Permutation Feature Importance — The Quick Global Picture

### The Core Idea

Permutation importance is the simplest of the three methods, and it's a great starting point. The intuition is almost elegant in how obvious it is:

> *If a feature truly matters, scrambling it should hurt the model. If it doesn't matter, scrambling it should change nothing.*

Think of a football team. If you secretly swap your striker with a random person from the crowd, the team will fall apart. But if you swap the backup goalkeeper's water bottle with a different one, nothing changes. The striker mattered. The water bottle didn't.

Here's how it works in practice:

1. Train your model. Record its accuracy (or R², or whatever metric you care about) on a validation set.
2. Pick one feature column — say, "Income" — and randomly shuffle its values across all the rows. Now Income is a mess: row 1's income belongs to row 47, row 2's belongs to row 203, etc.
3. Run the scrambled data through the model. Record the new accuracy.
4. The *drop* in accuracy is the importance score for Income.
5. Restore the column. Repeat for every other feature.

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Example: loan approval dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)

result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

print(importance_df)
```

A real result might look like this:

| Feature | Importance (R² drop) |
|---|---|
| House Size | 0.30 ★ Most important |
| Neighbourhood | 0.17 |
| No. of Rooms | 0.13 |
| House Age | 0.05 |
| Floor Number | 0.01 — Barely matters |

### The Catch: Correlated Features

Permutation importance has one serious flaw that you need to know about before using it.

When two features are correlated — like House Size and Number of Rooms (bigger houses tend to have more rooms) — shuffling one of them creates *unrealistic data*. Suddenly you have a 200 m² apartment with 1 room, or a 50 m² flat with 10 rooms. The model has never seen inputs like these in training. It gets confused, performs poorly, and you conclude that Number of Rooms is "important" — when really, you just broke the data.

Hooker & Mentch (2019) wrote a paper literally titled *"Please Stop Permuting Features"* to warn about this exact problem.[^1] The method is fine as a quick overview, but when features are correlated, the numbers can be misleading.

**When to use it:** As a fast, global first look — which features are worth investigating further?  
**When not to use it:** As your final word, especially with correlated features or when you need local (per-person) explanations.

---

## 3. LIME — Explaining One Decision at a Time

### The Core Idea

Permutation importance gives you a bird's-eye view: which features matter *in general*, across the whole dataset. But what if you want to know why *this specific person* was denied, not the average person?

That's where LIME comes in. LIME stands for **Local Interpretable Model-agnostic Explanations**, introduced by Ribeiro, Singh & Guestrin at KDD 2016 [^2], and the name is almost a complete description of the method:

- **Local** — it explains one prediction at a time
- **Interpretable** — the explanation is a simple, readable model
- **Model-agnostic** — it works with *any* black box, whether it's a neural network, a random forest, or anything else

The key insight in LIME comes from an analogy about the Earth. Globally, the Earth is complicated — mountains, oceans, cliffs, curves. But if you look right at your feet, the ground is basically flat. You can describe it with a simple equation. LIME applies the same logic: even the most complex neural network behaves *approximately linearly* in a very small neighbourhood around any single input.

So LIME zooms in, approximates the model locally with a simple line, and reads off the explanation from that line.

### How LIME Works — Step by Step

Let's use a concrete example: a loan applicant. Age 25, income 3,000 EGP, debt 8,000 EGP, unstable employment. The model says: **Denied (78% confidence)**.

**Step 1 — Generate Neighbours**  
LIME creates about 1,000 slightly modified versions of this applicant. Tweak the income a bit. Change the debt. Vary the age. These "neighbours" live in a small region around the original input.

**Step 2 — Query the Model**  
Feed all 1,000 neighbours to the actual black-box model. For each one, get a prediction: Approved or Denied, with probabilities.

**Step 3 — Weight by Distance**  
Neighbours that are very similar to the original applicant get higher weight. Neighbours that are very different count for less. This keeps the explanation *local* — we're not trying to understand the whole model, just the region around this one person.

**Step 4 — Fit a Linear Model**  
Train a simple weighted linear regression on the 1,000 labelled neighbours. This is the "interpretable" part — a linear model is easy to understand.

**Step 5 — Read the Weights**  
The coefficients of that linear model are the explanation. Which features pushed hardest toward denial?

```python
import lime
import lime.lime_tabular

# Create the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Approved', 'Denied'],
    mode='classification'
)

# Explain a single prediction
i = 42  # the applicant we want to explain
explanation = explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=5
)

explanation.show_in_notebook()
```

The output might look like this:

```
Debt (8,000 EGP)         → -0.41  (pushes toward Denied)
Employment (Unstable)    → -0.32
Credit History (Poor)    → -0.22
Income (3,000 EGP)       → +0.18  (pushes toward Approved)
Age (25)                 → +0.05
```

**Actionable insight:** reducing debt has the biggest single impact on approval. The model is essentially saying: fix your debt situation first.

### LIME on Text Data

One of the nicest things about LIME is that it works on more than just numbers. For text data, instead of tweaking numerical values, LIME removes words one at a time and sees which ones change the prediction.

Take a restaurant review: *"The food was amazing but the service was terrible and the staff were rude."*  
The model predicts: **NEGATIVE (71% confidence).**

LIME's word-level explanation:

```
terrible   → -0.38  (strongly negative)
rude       → -0.28
amazing    → +0.31  (strongly positive, but outvoted)
food       → +0.04
```

The model understood that "amazing food" pulled the review toward positive, but "terrible service" and "rude staff" were stronger. It got the right answer for the right reasons.

```python
from lime.lime_text import LimeTextExplainer

text_explainer = LimeTextExplainer(class_names=['Positive', 'Negative'])
review = "The food was amazing but the service was terrible and the staff were rude."

exp = text_explainer.explain_instance(
    review,
    text_model.predict_proba,
    num_features=6
)

exp.show_in_notebook()
```

### LIME's Limitations

LIME is practical and widely used, but it has real weaknesses:

**Instability:** Run LIME twice on the same input, and you might get different explanations. Because the neighbours are sampled randomly, there's variance in the result. This is a known problem — Slack et al. (2020) [^3] documented it in detail.

**The neighbourhood problem:** How wide should "local" be? Too wide and the linear approximation is wrong. Too narrow and you don't have enough samples. There's no principled way to choose, and the default settings don't always make sense.

**Correlated features:** Just like with permutation, LIME can generate unrealistic neighbours when features are correlated.

**No global view:** LIME explains one prediction. Aggregating many LIME explanations into a global picture is not straightforward.

---

## 4. SHAP — Fair Credit for Every Feature

### Where SHAP Comes From

SHAP (SHapley Additive exPlanations) was introduced by Lundberg & Lee at NeurIPS 2017 [^4], and it's grounded in something much older: game theory from the 1950s.

Here's the problem that motivated it. Three people — let's call them Ali, Badr, and Sara — start a company together. At the end of the year, they've made 90,000 EGP. How do you divide that fairly? Ali was involved in everything. Badr joined a specific project. Sara came in late. Who gets what?

Lloyd Shapley solved this problem in 1953 (and won the Nobel Prize in Economics in 2012 for related work). His solution: for each person, look at *every possible order* in which they could have joined the team. Measure how much extra profit each person added each time they joined. Average those marginal contributions across all orderings. That average is their fair share — the Shapley value.

SHAP applies this to machine learning:
- **Players** = features
- **Profit** = the model's prediction
- **Shapley value** = each feature's fair contribution to the prediction

### The Key Guarantee

What makes SHAP special is that the values always add up exactly to the prediction. This isn't an approximation — it's mathematically guaranteed.

For our loan applicant (base rate: 60% approval, final prediction: 20%):

```
Base rate (dataset average)    60%
+ SHAP(Debt: 8,000 EGP)       -25%  →  35%
+ SHAP(Employment: Unstable)  -10%  →  25%
+ SHAP(Age: 25)                -8%  →  17%
+ SHAP(Loan amount: 50K)       -4%  →  13%
+ SHAP(Income: 3,000 EGP)      +3%  →  16%
+ SHAP(Other features)         +4%  →  20%
= Final prediction              20%  ✓
```

Every decimal point is accounted for. Nothing is hidden. LIME has no equivalent guarantee — its explanations are approximations.

### Three Mathematical Axioms

SHAP is the only attribution method that satisfies all three of these simultaneously:

**Axiom 1 — Local Accuracy:** SHAP values always sum exactly to the model's prediction. Nothing is unexplained.

**Axiom 2 — Missingness:** A feature that is absent from the input always gets a SHAP value of exactly zero. No phantom contributions from things the model never saw.

**Axiom 3 — Consistency:** If a feature becomes more important in a new version of the model, its SHAP value never decreases. Explanations remain coherent as models change.

### Using SHAP in Practice

```python
import shap

# For tree-based models (fast — uses TreeSHAP)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# For any model (slower — uses KernelSHAP)
# explainer = shap.KernelExplainer(model.predict_proba, X_train_sample)
# shap_values = explainer.shap_values(X_test)

# Explain one prediction
shap.plots.waterfall(shap_values[0])

# Summarize across the whole dataset
shap.plots.beeswarm(shap_values)
```

### SHAP Visualizations

SHAP comes with a suite of visualizations that are genuinely among the best in the field.

**Force Plot — One Prediction**  
Shows which features pushed a single prediction up or down from the baseline. Red bars push toward denial; teal bars push toward approval. The width of each bar is proportional to the feature's SHAP value.

```python
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0])
```

**Waterfall Plot — Step-by-Step Walkthrough**  
Shows how the prediction builds up from the baseline, one feature at a time. Best for regulatory contexts or explaining decisions to customers — you can literally walk someone through the numbers.

```python
shap.plots.waterfall(shap_values[0])
```

**Beeswarm Plot — Whole Dataset at Once**  
Each dot is one person in the dataset. The horizontal position shows the SHAP value (positive = pushed toward approval). The colour shows the feature's raw value (red = high, blue = low). The spread of the dots shows overall importance. This is how you go from local to global.

```python
shap.plots.beeswarm(shap_values)
```

A beeswarm for the loan dataset might show: high debt (red dots, far left) consistently hurts approval, while higher income (red dots, far right) consistently helps.

### SHAP's Limitations

SHAP is powerful, but not perfect.

**Computational cost:** Exact SHAP requires averaging over all possible feature orderings — that's exponential in the number of features. For tree-based models, TreeSHAP (Lundberg et al., 2018) [^5] solves this with a polynomial-time algorithm. For deep learning, it's still expensive.

**Correlated features:** SHAP still has to deal with unrealistic inputs when features are correlated, though it handles this better than the other methods.

**It can be fooled:** Slack et al. (2020) [^3] showed that a model can be designed to show fair-looking SHAP values on testing data while actually discriminating in production. Explanations are only as honest as the model they're explaining.

---

## 5. Real-World Case Studies

### Case Study 1: Credit Scoring (FICO Challenge, 2018)

The FICO Explainable Machine Learning Challenge [^6] tasked participants with not just predicting credit defaults, but *explaining every individual decision*. SHAP was central to the winning approaches.

In this context, GDPR compliance required per-applicant explanations for every denial. SHAP made it possible to tell someone: "Your approval probability dropped 25% because your debt is in the top 10% of applicants."

But the most interesting finding wasn't in individual predictions. Aggregating SHAP values across the dataset revealed that "number of bank calls" — how often an applicant had called the bank — was a strong predictor of default. That sounds reasonable until you think about it: people call the bank when they're in financial distress. The model was using distress as a predictor of default, which creates a feedback loop that punishes people for seeking help. Once identified via SHAP, the feature was removed.

Zip codes also emerged as a predictive feature — a potential proxy for demographic information. The SHAP analysis flagged it for investigation before deployment.

### Case Study 2: COVID-19 Triage (Yan et al., 2020)

Yan et al. (2020) [^7] built an XGBoost model to predict COVID-19 mortality from routine blood tests, trained on 375 patients in Wuhan. The model achieved over 90% accuracy — exceeding the baseline performance of ICU physicians.

The model was accurate. But doctors refused to use it. "I won't act on a black box" is a completely reasonable position when lives are on the line.

TreeSHAP changed that. For each patient, the model could now explain: "LDH of 520 contributed +35% mortality risk because it is critically elevated — above the threshold associated with significant tissue damage." The doctors could look at the SHAP values and verify that the model was using the same clinical markers they would use.

The result: the model was adopted into clinical practice. Accuracy alone wasn't enough. Explainability was the deciding factor.

The three top SHAP features were:
1. **LDH (lactate dehydrogenase)** — a marker of tissue damage
2. **Lymphocyte count** — measures immune response
3. **High-sensitivity CRP** — an inflammation marker

All three are clinically meaningful. SHAP confirmed the model was doing the right thing for the right reasons.

### Case Study 3: Fake News Detection with LIME

Ribeiro et al.'s original LIME paper included a text classification example [^2] that's still one of the clearest demonstrations of why feature-level explanations matter for text models.

Consider a news classifier given this article excerpt: *"Scientists CONFIRM that vaccines CAUSE harm. Experts say the research was rigorous but the sensational headline misrepresents it."*

The model says: **UNRELIABLE (83% confidence).**

But why? Without an explanation, the classifier is useless for editors who need to make decisions. LIME's word-level output might reveal that "CONFIRM," "CAUSE," and the all-caps formatting drove the classification — not the nuanced second sentence. That's useful: it tells you the model is picking up on sensationalist language patterns, which is exactly what you'd want a fake news detector to do.

It also tells you something else: the model might be wrong on articles that happen to use emphatic language for legitimate reasons. That's worth knowing before deployment.

---

## 6. Comparison: Permutation vs. LIME vs. SHAP

| | Permutation | LIME | SHAP |
|---|---|---|---|
| **Scope** | Global only | Local only | Local + Global |
| **Guarantee** | None | None | Sums to prediction |
| **Speed** | Fast | Medium | Slow (except TreeSHAP) |
| **Works on any model** | Yes | Yes | Yes (KernelSHAP) |
| **Handles correlated features** | Poorly | Poorly | Better, still not perfect |
| **Visualizations** | Bar chart | Weights per feature | Force, Beeswarm, Waterfall |
| **Best for** | Quick overview | Fast local diagnosis | Rigorous, regulatory |

The practical advice: **use all three together.** Permutation gives you the big picture. LIME gives you a fast, local diagnosis for individual predictions. SHAP gives you rigorous, mathematically guaranteed explanations for anything that matters.

---

## 7. The Shared Limit: Correlation Is Not Causation

This is the most important thing to take away from this chapter.

All three methods explain what the *model* associated with the prediction — not what *actually caused the outcome* in reality. If a model learned that people with black hair have higher loan default rates (because of some spurious correlation in the training data), SHAP will dutifully report "black hair: -15% approval probability." The explanation is honest about the model. It says nothing about the world.

This distinction is critical in healthcare, criminal justice, and any domain where decisions have serious consequences. Rudin (2019) [^8] argues that for high-stakes decisions, we shouldn't be using opaque models with post-hoc explanations at all — we should be building *inherently interpretable models* (decision trees, rule lists, linear models) that are transparent by design. Post-hoc explanations, she argues, are a band-aid on a wound that needs surgery.

Additionally, Slack et al. (2020) [^3] showed that it's possible to engineer a model that produces fair-looking SHAP and LIME explanations on a test set while discriminating in production. If someone has enough motivation to game the explanation system, they can.

Feature-based explanations are powerful. They've helped catch biased models before deployment, enabled clinical adoption of AI, and given regulators the tools they need. But they're a starting point, not an ending point. Always validate with domain experts. Always ask whether the model's associations reflect causal reality. And always wonder whether the explanation is telling you the truth about the model — or just what the model has learned to say.

---

## 8. Try It Yourself

If you want to experiment with these methods on your own:

- **SHAP library:** [shap.readthedocs.io](https://shap.readthedocs.io)
- **LIME library:** [github.com/marcotcr/lime](https://github.com/marcotcr/lime)
- **Molnar's free book** (highly recommended for going deeper): [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)
- **FICO Challenge dataset and notebooks:** [community.fico.com/s/explainable-machine-learning-challenge](https://community.fico.com/s/explainable-machine-learning-challenge)

A good exercise: take any classification dataset you've worked with before, train a random forest, and try explaining a single prediction with both LIME and SHAP. Notice where they agree. Notice where they don't. Ask yourself why.

---

## References

[^1]: Hooker, S., & Mentch, L. (2019). *Please Stop Permuting Features: An Explanation and Alternatives.* arXiv:1905.03151. [arxiv.org/abs/1905.03151](https://arxiv.org/abs/1905.03151)

[^2]: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier.* ACM KDD. arXiv:1602.04938. [arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938)

[^3]: Slack, D., et al. (2020). *Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods.* AAAI AIES. arXiv:1911.02508. [arxiv.org/abs/1911.02508](https://arxiv.org/abs/1911.02508)

[^4]: Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. arXiv:1705.07874. [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)

[^5]: Lundberg, S. M., et al. (2018). *Consistent Individualized Feature Attribution for Tree Ensembles.* arXiv:1802.03888. [arxiv.org/abs/1802.03888](https://arxiv.org/abs/1802.03888)

[^6]: FICO Explainable Machine Learning Challenge (2018). [community.fico.com/s/explainable-machine-learning-challenge](https://community.fico.com/s/explainable-machine-learning-challenge)

[^7]: Yan, L., et al. (2020). *An interpretable mortality prediction model for COVID-19 patients.* Nature Machine Intelligence 2, 283–295. [nature.com/articles/s42256-020-0180-7](https://www.nature.com/articles/s42256-020-0180-7)

[^8]: Rudin, C. (2019). *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.* Nature Machine Intelligence. arXiv:1811.10154. [arxiv.org/abs/1811.10154](https://arxiv.org/abs/1811.10154)

[^9]: Fisher, A., Rudin, C., & Dominici, F. (2019). *All Models are Wrong, but Many are Useful.* JMLR 20(177). arXiv:1801.01489. [arxiv.org/abs/1801.01489](https://arxiv.org/abs/1801.01489)

[^10]: Molnar, C. (2022). *Interpretable Machine Learning (2nd ed.).* [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book)

---

*To cite this chapter, please use the following BibTeX:*

```bibtex
@misc{hemaya_2026_XAI,
  author       = {Youssef Hemaya},
  title        = {Interpreting Machine Learning: A Gentle Introduction, Chapter 3},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/interpreting_machine_learning}},
}
```
