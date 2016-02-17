package lc.advanced;

import annotation.DissimilarityWeight;
import annotation.RatingPredictor;
import annotation.RelevanceWeight;
import annotation.UnpopWeight;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;
import pop.PopModel;
import util.AlgorithmUtil;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;

public class LCAdvancedItemScorer extends AbstractItemScorer {
	private PopModel model;
	private final ItemScorer itemScorer;
	private final PreferenceSnapshot snapshot;
	private final LCModel lcModel;
	private final PreferenceDomain domain;

	@Inject
	public LCAdvancedItemScorer(PopModel model, @RatingPredictor ItemScorer itemScorer,
								@Transient @Nonnull PreferenceSnapshot snapshot, PreferenceDomain domain, LCModel lcModel) {
		this.model = model;
		this.itemScorer = itemScorer;
		this.snapshot = snapshot;
		this.domain = domain;
		this.lcModel = lcModel;
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		for (VectorEntry e : scores.view(VectorEntry.State.EITHER)) {
			double dissim = 0;
			dissim = getDissim(user, e.getKey());
			double unpop = 1.0 - (double) model.getPop(e.getKey()) / model.getMax();
			double rating = itemScorer.score(user, e.getKey());
			double total = lcModel.getUw() * unpop + lcModel.getRw() * rating / domain.getMaximum() + lcModel.getDw() * dissim;
			scores.set(e, total);
		}
	}

	private double getDissim(long userId, long itemId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		SparseVector itemVec = AlgorithmUtil.itemContentMap.get(itemId);
		double dissim = 0.0;
		for (IndexedPreference p : prefs) {
			SparseVector ratedItemVec = AlgorithmUtil.itemContentMap.get(p.getItemId());
			dissim += 1.0 - ContentUtil.getCosine(ratedItemVec, itemVec);
		}
		return dissim / prefs.size();
	}
}
