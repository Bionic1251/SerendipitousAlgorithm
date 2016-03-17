package lc.investigation;


import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;

import pop.PopModel;
import util.AverageAggregate;


import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.Collection;


public class NonPersInvestigationItemScorer extends InvestigationItemScorer {
	private double wr;
	private double wd;
	private double wu;
	private static final int ITERATION_COUNT = 5;

	@Inject
	public NonPersInvestigationItemScorer(PopModel popModel, @Transient @Nonnull PreferenceSnapshot snapshot) {
		super(popModel, snapshot, ITERATION_COUNT);
	}

	@Override
	protected void init() {
		wr = DEFAULT_VAL;
		wd = DEFAULT_VAL;
		wu = DEFAULT_VAL;
	}

	@Override
	protected void trainForEachUser(long userId) {
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		AverageAggregate aggregate = userThresholdMap.get(userId);
		for (IndexedPreference innerPref : prefs) {
			for (IndexedPreference outerPref : prefs) {
				if (innerPref.getItemId() == outerPref.getItemId()) {
					continue;
				}
				Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue(), aggregate);
				Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue(), aggregate);
				boolean innerSer = isSerendipitous(innerTriple, aggregate);
				boolean outerSer = isSerendipitous(outerTriple, aggregate);
				if (!innerSer && outerSer) {
					changeParameters(outerTriple, innerTriple);
				} else {
					if (innerSer && !outerSer) {
						changeParameters(innerTriple, outerTriple);
					}
				}
			}
		}
	}

	@Override
	protected void printWeights() {
		System.out.println("Weights are wr=" + wr + "; wd=" + wd + "; wu=" + wu);
	}

	private void changeParameters(Triple serTriple, Triple unserTriple) {
		double pwr = 0, pwd = 0, pwu = 0;
		if (wr + wd + wu > 1) {
			pwr -= P_COEFF;
			pwd -= P_COEFF;
			pwu -= P_COEFF;
		}
		if (wr < 0) {
			pwr += P_COEFF;
		}
		if (wd < 0) {
			pwd += P_COEFF;
		}
		if (wu < 0) {
			pwu += P_COEFF;
		}
		double derR = serTriple.rating - unserTriple.rating;
		double derD = serTriple.dissimilarity - unserTriple.dissimilarity;
		double derU = serTriple.unpopularity - unserTriple.unpopularity;
		wr += LEARNING_RATE * derR + pwr;
		wd += LEARNING_RATE * derD + pwd;
		wu += LEARNING_RATE * derU + pwu;
	}

	private double getPredictedSerendipity(Triple triple) {
		double result = wr * triple.rating + wd * triple.dissimilarity + wu * triple.unpopularity;
		return result;
	}

	@Override
	protected double getFunctionVal(long userId) {
		AverageAggregate aggregate = userThresholdMap.get(userId);
		Collection<IndexedPreference> prefs = snapshot.getUserRatings(userId);
		double sum = 0;
		for (IndexedPreference innerPref : prefs) {
			for (IndexedPreference outerPref : prefs) {
				if (innerPref.getItemId() == outerPref.getItemId()) {
					continue;
				}
				Triple innerTriple = new Triple(userId, innerPref.getItemId(), innerPref.getValue(), aggregate);
				Triple outerTriple = new Triple(userId, outerPref.getItemId(), outerPref.getValue(), aggregate);
				boolean innerSer = isSerendipitous(innerTriple, aggregate);
				boolean outerSer = isSerendipitous(outerTriple, aggregate);
				double val = 0;
				if (!innerSer && outerSer) {
					val = getPredictedSerendipity(outerTriple) - getPredictedSerendipity(innerTriple);
				} else {
					if (innerSer && !outerSer) {
						val = getPredictedSerendipity(innerTriple) - getPredictedSerendipity(outerTriple);
					}
				}
				sum += val;
			}
		}
		return sum;
	}
}
