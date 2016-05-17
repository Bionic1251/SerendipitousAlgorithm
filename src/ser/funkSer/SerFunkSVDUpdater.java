package ser.funkSer;

import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.mf.funksvd.FunkSVDUpdateRule;

public class SerFunkSVDUpdater {
	private final SerFunkSVDUpdateRule updateRule;

	private double error;
	private double userFeatureValue;
	private double itemFeatureValue;
	private double sse;
	private int n;

	SerFunkSVDUpdater(SerFunkSVDUpdateRule rule) {
		updateRule = rule;
	}

	/**
	 * Reset the statistics and counters tracked by this updater.
	 */
	public void resetStatistics() {
		sse = 0;
		n = 0;
	}

	/**
	 * Get the number of updates this updater has prepared since the last reset.
	 *
	 * @return The number of updates done.
	 * @see #resetStatistics()
	 */
	public int getUpdateCount() {
		return n;
	}

	/**
	 * Get the RMSE of all updates done since the last reset.
	 *
	 * @return The root-mean-squared error of the updates since the last reset.
	 */
	public double getRMSE() {
		if (n <= 0) {
			return Double.NaN;
		} else {
			return Math.sqrt(sse / n);
		}
	}

	/**
	 * Prepare the updater for updating the feature values for a particular user/item ID.
	 *
	 * @param feature  The feature we are training.
	 * @param rating   The rating value.
	 * @param estimate The estimate through the previous feature.
	 * @param uv       The user feature value.
	 * @param iv       The item feature value.
	 * @param trail    The sum of the trailing feature value products.
	 */

	private double w;
	public void prepare(int feature, double rating, double estimate,
						double uv, double iv, double trail, double w) {
		this.w = w;
		// Compute prediction
		double pred = estimate + uv * iv;
		PreferenceDomain dom = updateRule.getDomain();
		if (dom != null) {
			pred = dom.clampValue(pred);
		}
		pred += trail;

		// Compute the err and store this value
		error = rating - pred;
		userFeatureValue = uv;
		itemFeatureValue = iv;

		// Update statistics
		n += 1;
		sse += error * error;
	}

	/**
	 * Get the error from the prepared update.
	 *
	 * @return The estimation error in the prepared update.
	 */
	public double getError() {
		return error;
	}

	/**
	 * Get the update for the user-feature value.
	 *
	 * @return The delta to apply to the user-feature value.
	 */
	public double getUserFeatureUpdate() {
		double delta = error * itemFeatureValue - updateRule.getTrainingRegularization() * userFeatureValue;
		return delta * updateRule.getLearningRate();
	}

	/**
	 * Get the update for the item-feature value.
	 *
	 * @return The delta to apply to the item-feature value.
	 */
	public double getItemFeatureUpdate() {
		double delta = error * w * userFeatureValue - updateRule.getTrainingRegularization() * itemFeatureValue;
		return delta * updateRule.getLearningRate();
	}
}
