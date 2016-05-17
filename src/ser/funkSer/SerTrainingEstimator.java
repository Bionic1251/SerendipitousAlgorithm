package ser.funkSer;

import it.unimi.dsi.fastutil.longs.LongCollection;
import it.unimi.dsi.fastutil.longs.LongIterator;
import mikera.vectorz.AVector;
import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;

import java.util.Collection;

public class SerTrainingEstimator {
	private final Collection<IndexedPreference> ratings;
	private final double[] estimates;
	private final PreferenceDomain domain;

	/**
	 * Initialize the training estimator.
	 *
	 * @param snap     The preference snapshot.
	 * @param baseline The genrePredictor predictor.
	 * @param dom      The preference domain (for clamping).
	 */
	SerTrainingEstimator(PreferenceSnapshot snap, ItemScorer baseline, PreferenceDomain dom) {
		ratings = snap.getRatings();
		domain = dom;
		estimates = new double[ratings.size()];

		final LongCollection userIds = snap.getUserIds();
		LongIterator userIter = userIds.iterator();
		while (userIter.hasNext()) {
			long uid = userIter.nextLong();
			SparseVector rvector = snap.userRatingVector(uid);
			MutableSparseVector blpreds = MutableSparseVector.create(rvector.keySet());
			baseline.score(uid, blpreds);

			for (IndexedPreference r : snap.getUserRatings(uid)) {
				estimates[r.getIndex()] = blpreds.get(r.getItemId());
			}
		}
	}

	/**
	 * Get the estimate for a preference.
	 *
	 * @param pref The preference.
	 * @return The estimate.
	 */
	public double get(IndexedPreference pref) {
		return estimates[pref.getIndex()];
	}

	/**
	 * Update the current estimates with trained values for a new feature.
	 *
	 * @param ufvs The user feature values.
	 * @param ifvs The item feature values.
	 */
	public void update(AVector ufvs, AVector ifvs) {
		for (IndexedPreference r : ratings) {
			int idx = r.getIndex();
			double est = estimates[idx];
			est += ufvs.get(r.getUserIndex()) * ifvs.get(r.getItemIndex());
			if (domain != null) {
				est = domain.clampValue(est);
			}
			estimates[idx] = est;
		}
	}
}

