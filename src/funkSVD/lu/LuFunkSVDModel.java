package funkSVD.lu;

import com.google.common.collect.ImmutableList;
import mikera.matrixx.impl.ImmutableMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.impl.ImmutableVector;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.grouplens.lenskit.core.Shareable;
import org.grouplens.lenskit.indexes.IdIndexMapping;
import org.grouplens.lenskit.mf.funksvd.FeatureInfo;
import org.grouplens.lenskit.mf.svd.MFModel;

import java.util.List;

@DefaultProvider(LuFunkSVDModelBuilder.class)
@Shareable
public final class LuFunkSVDModel extends MFModel {
	private final List<FeatureInfo> featureInfo;
	private final AVector averageUser;

	public LuFunkSVDModel(ImmutableMatrix umat, ImmutableMatrix imat,
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

	public List<FeatureInfo> getFeatureInfo() {
		return featureInfo;
	}

	public AVector getAverageUserVector() {
		return averageUser;
	}
}
