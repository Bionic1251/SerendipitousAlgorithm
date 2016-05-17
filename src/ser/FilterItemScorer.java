package ser;

import annotation.RatingPredictor;
import annotation.RatingPredictor2;
import evaluationMetric.Container;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import pop.PopModel;
import util.Settings;
import util.Util;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

public class FilterItemScorer extends AbstractItemScorer {
	private final ItemScorer itemScorer;
	private final ItemScorer obviousItemScorer;
	private final Set<Long> itemSet;
	private final PreferenceSnapshot snapshot;

	@Inject
	public FilterItemScorer(@RatingPredictor ItemScorer itemScorer, PopModel popModel, @RatingPredictor2 ItemScorer obviousItemScorer, PreferenceSnapshot snapshot) {
		this.itemScorer = itemScorer;
		this.obviousItemScorer = obviousItemScorer;
		this.snapshot = snapshot;
		List<Long> list = new ArrayList<Long>(popModel.getItemList());
		Collections.reverse(list);
		itemSet = new HashSet<Long>(list.subList(0, Settings.POPULAR_ITEMS_SERENDIPITY_NUMBER));
	}

	@Override
	public void score(long user, @Nonnull MutableSparseVector scores) {
		MutableSparseVector vector = MutableSparseVector.create(snapshot.getItemIds());
		obviousItemScorer.score(user, vector);
		Set<Long> expectedSet = Util.getExpectedSet(user, vector, snapshot);
		itemScorer.score(user, scores);
		for (Long key : scores.keySet()) {
			if (itemSet.contains(key) || expectedSet.contains(key)) {
				scores.set(key, 0.0);
			}
		}
	}
}
