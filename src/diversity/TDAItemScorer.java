package diversity;

import annotation.Alpha;
import annotation.RatingPredictor;
import evaluationMetric.Container;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.basic.AbstractItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import util.ContentAverageDissimilarity;
import util.ContentUtil;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

public class TDAItemScorer extends TDABasic {
	private double factor;

	@Inject
	public TDAItemScorer(@RatingPredictor ItemScorer itemScorer, PreferenceSnapshot snapshot, @Alpha double factor) {
		super(itemScorer, snapshot);
		this.factor = factor;
		System.out.println("TDAItemScorer factor=" + factor);
	}

	@Override
	protected List<Container<Double>> mixLists(List<Container<Double>> relevantCandidates, List<Container<Double>> diversifiedCandidates, long userId) {
		List<Container<Double>> scoreList = new ArrayList<Container<Double>>();
		int maxPos = relevantCandidates.size();
		for (Container<Double> item : relevantCandidates) {
			double relevantRank = maxPos - relevantCandidates.indexOf(item);
			double diversifiedRank = maxPos - diversifiedCandidates.indexOf(item);
			double score = relevantRank * (1 - factor) + diversifiedRank * factor;
			scoreList.add(new Container<Double>(item.getId(), score));
		}
		Collections.sort(scoreList);
		Collections.reverse(scoreList);
		return scoreList;
	}
}
