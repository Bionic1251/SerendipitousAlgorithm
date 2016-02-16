package mf.baseline;

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

public class SVDModelBuilder implements Provider<SVDModel> {
	private static Logger logger = LoggerFactory.getLogger(SVDModelBuilder.class);

	protected final int featureCount;
	protected final double learningRate;
	protected final double regularization;
	private final StoppingCondition stoppingCondition;
	protected final PreferenceSnapshot snapshot;
	protected final double initialValue;

	@Inject
	public SVDModelBuilder(@Transient @Nonnull PreferenceSnapshot snapshot,
						   @FeatureCount int featureCount,
						   @InitialFeatureValue double initVal, @LearningRate double lrate,
						   @RegularizationTerm double reg, StoppingCondition stop) {
		this.featureCount = featureCount;
		this.initialValue = initVal;
		this.snapshot = snapshot;
		learningRate = lrate;
		regularization = reg;
		stoppingCondition = stop;
	}


	@Override
	public SVDModel get() {
		System.out.println(SVDModelBuilder.class);
		int userCount = snapshot.getUserIds().size();
		Matrix userFeatures = Matrix.create(userCount, featureCount);

		int itemCount = snapshot.getItemIds().size();
		Matrix itemFeatures = Matrix.create(itemCount, featureCount);

		logger.info("Building baseline with {} features for {} ratings",
				featureCount, snapshot.getRatings().size());

		List<FeatureInfo> featureInfo = new ArrayList<FeatureInfo>(featureCount);

		Vector uvec = Vector.createLength(userCount);
		Vector ivec = Vector.createLength(itemCount);

		for (int f = 0; f < featureCount; f++) {
			uvec.fill(initialValue);
			ivec.fill(initialValue);

			userFeatures.setColumn(f, uvec);
			itemFeatures.setColumn(f, ivec);

			FeatureInfo.Builder fib = new FeatureInfo.Builder(f);
			featureInfo.add(fib.build());
		}

		TrainingLoopController controller = stoppingCondition.newLoop();
		calculateStatistics(userFeatures, itemFeatures);
		while (controller.keepTraining(0.0)) {
			trainFeatures(userFeatures, itemFeatures);
			calculateStatistics(userFeatures, itemFeatures);
		}

		return new SVDModel(ImmutableMatrix.wrap(userFeatures),
				ImmutableMatrix.wrap(itemFeatures),
				snapshot.userIndex(), snapshot.itemIndex(),
				featureInfo);
	}

	private void trainFeatures(Matrix userFeatures, Matrix itemFeatures) {
		for (IndexedPreference rating : snapshot.getRatings()) {
			AVector item = itemFeatures.getRow(rating.getItemIndex());
			AVector user = userFeatures.getRow(rating.getUserIndex());
			double prediction = item.dotProduct(user);
			double error = (rating.getValue() - prediction);
			for (int i = 0; i < featureCount; i++) {
				double val = item.get(i) + learningRate * (2 * error * user.get(i) - regularization * item.get(i));
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println(val);
				}
				item.set(i, val);

				val = user.get(i) + learningRate * (2 * error * item.get(i) - regularization * user.get(i));
				if (Double.isNaN(val) || Double.isInfinite(val)) {
					System.out.println(val);
				}
				user.set(i, val);
			}
		}
	}

	private void calculateStatistics(Matrix userFeatures, Matrix itemFeatures) {
		double sum = 0;
		for (IndexedPreference rating : snapshot.getRatings()) {
			AVector item = itemFeatures.getRow(rating.getItemIndex());
			AVector user = userFeatures.getRow(rating.getUserIndex());
			double prediction = item.dotProduct(user);
			double error = (rating.getValue() - prediction);
			sum += Math.abs(error);
		}
		System.out.println("MAE " + sum / snapshot.getRatings().size());
	}
}
