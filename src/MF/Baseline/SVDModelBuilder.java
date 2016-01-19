package MF.Baseline;

import annotation.Alpha;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.grouplens.lenskit.core.Transient;
import org.grouplens.lenskit.data.pref.IndexedPreference;
import org.grouplens.lenskit.data.pref.PreferenceDomain;
import org.grouplens.lenskit.data.snapshot.PreferenceSnapshot;
import org.grouplens.lenskit.iterative.LearningRate;
import org.grouplens.lenskit.iterative.RegularizationTerm;
import org.grouplens.lenskit.iterative.StoppingCondition;
import org.grouplens.lenskit.iterative.TrainingLoopController;
import org.grouplens.lenskit.mf.funksvd.FeatureCount;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.InitialFeatureValue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.inject.Inject;
import javax.inject.Provider;
import java.util.*;

/**
 * Baseline recommender builder using gradient descent (Funk Baseline).
 * <p/>
 * <p>
 * This recommender builder constructs an Baseline-based recommender using gradient
 * descent, as pioneered by Simon Funk.  It also incorporates the regularizations
 * Funk did. These are documented in
 * <a href="http://sifter.org/~simon/journal/20061211.html">Netflix Update: Try
 * This at Home</a>. This implementation is based in part on
 * <a href="http://www.timelydevelopment.com/demos/NetflixPrize.aspx">Timely
 * Development's sample code</a>.</p>
 *
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SVDModelBuilder implements Provider<SVDModel> {
	private static Logger logger = LoggerFactory.getLogger(SVDModelBuilder.class);

	protected final int featureCount;
	protected final double learningRate;
	protected final double regularization;
	private final PreferenceDomain domain;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;

	@Inject
	public SVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
						   @FeatureCount int featureCount,
						   @InitialFeatureValue double initVal,
						   @Nullable PreferenceDomain dom, @LearningRate double lrate,
						   @RegularizationTerm double reg, StoppingCondition stop) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		domain = dom;
		learningRate = lrate;
		regularization = reg;
		stoppingCondition = stop;
	}


	@Override
	public SVDModel get() {
		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.info("Building Baseline with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			uvec.fill(initialValue);
			ivec.fill(initialValue);

			userFeatures.setColumn(f, uvec);
			assert Math.abs(userFeatures.getColumnView(f).elementSum() - uvec.elementSum()) < 1.0e-4 : "user column sum matches";
			itemFeatures.setColumn(f, ivec);
			assert Math.abs(itemFeatures.getColumnView(f).elementSum() - ivec.elementSum()) < 1.0e-4 : "item column sum matches";

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			featureInfo.add(fib.build());
		}

		TrainingLoopController controller = stoppingCondition.newLoop();
		while (controller.keepTraining(0.0)) {
			trainFeatures(userFeatures, itemFeatures);
		}

		// Wrap the user/item matrices because we won't use or modify them again
		return new SVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void trainFeatures(Matrix userFeatures, Matrix itemFeatures) {
		double sum = 0;
		for (IndexedPreference rating : snapshot.getRatings()) {
			AVector item = itemFeatures.getRow(rating.getItemIndex());
			AVector user = userFeatures.getRow(rating.getUserIndex());
			double prediction = item.dotProduct(user);
			double error = (rating.getValue() - prediction);
			if (Double.isNaN(error) || Double.isInfinite(error)) {
				System.out.printf("Yo");
			}
			sum += Math.abs(error);
			for (int i = 0; i < featureCount; i++) {
				double val = item.get(i) + learningRate * (2 * error * user.get(i) - regularization * item.get(i));
				if (item.get(i) > 100 || user.get(i) > 100) {
					System.out.println("item " + item.get(i));
					System.out.println("user " + user.get(i));
				}
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println("NaN");
				}
				item.set(i, val);

				val = user.get(i) + learningRate * (2 * error * item.get(i) - regularization * user.get(i));
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println("NaN");
				}
				user.set(i, val);
			}
			System.out.println("MAE " + sum / snapshot.getRatings().size());
		}
	}
}
