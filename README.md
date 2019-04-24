# Master's Thesis

> Conversation-aware Classification of News Comments

By [Johannes Filter](https://johannesfilter.com/) at the [Web Science Group](https://hpi.de/naumann/web-science-group/info.html) at the Hasso-Plattner-Institue (University of Potsdam) under the supervision of [Dr. Ralf Krestel](https://hpi.de/naumann/sites/krestel/).

## Abstract

Online newspapers close their comment section because they cannot cope with the sheer amount of user-generated content. Natural-language processing allows to automatically classify news comments in order to efficiently support moderators. Identifying hate speech is only a special case of comment classification and in this master's thesis we focus on classifying along any classification criteria, e.g., sentiment, off-topic, controversial. In contrast to prior work, we consider the conversational context to be essential for understanding a comment's true meaning. We introduce a preprocessing technique to prepend previous comments to training samples in order to apply state-of-the-art language-model-based text classification technique ULMFIT. We conducted experiments on nine categories of the research dataset Yahoo News Annotated Comment Corpus. With conversation-aware models, we increased the F1 micro and F1 macro scores on average by 1.53% and 3.08%, respectively. However, the differences to conversation-agnostic models vary among the categories. We achieved the biggest improvements when identifying whether a comment is off-topic or if it agrees or disagrees with other comments.

## Caveats

This repository contains all the used code but it's rather unstructured. Nevertheless, maybe it can helpful for your work.

## License

MIT.
