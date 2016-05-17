package ser.funkSer;

import com.google.common.collect.ImmutableList;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.impl.ImmutableVector;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.indexes.IdIndexMapping;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.funksvd.FunkSVDModelBuilder;
import org.grouplens.lenskit.mf.svd.MFModel;

import java.util.List;

@DefaultProvider(SerFunkSVDModelBuilder.class)
@Shareable
public class SerFunkSVDModel extends MFModel {
	private static final long serialVersionUID = 3L;

	private final List<FeatureInfo> featureInfo;
	private final AVector averageUser;

	public SerFunkSVDModel(ImmutableMatrix umat, ImmutableMatrix imat,
						IdIndexMapping uidx, IdIndexMapping iidx,
						List<FeatureInfo> features) {
		super(umat, imat, uidx, iidx);

		featureInfo = ImmutableList.copyOf(features);

		double[] means = new double[featureCount];
		for (int f = featureCount - 1; f >= 0; f--) {
			means[f] = featureInfo.get(f).getUserAverage();
		}
		averageUser = ImmutableVector.wrap(means);
	}

	/**
	 * Get the {@link FeatureInfo} for a particular feature.
	 *
	 * @param f The feature number.
	 * @return The feature's summary information.
	 */
	public FeatureInfo getFeatureInfo(int f) {
		return featureInfo.get(f);
	}

	/**
	 * Get the metadata about all features.
	 *
	 * @return The feature metadata.
	 */
	public List<FeatureInfo> getFeatureInfo() {
		return featureInfo;
	}

	public AVector getAverageUserVector() {
		return averageUser;
	}

}
