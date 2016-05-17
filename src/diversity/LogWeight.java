package diversity;

import diversity.TDAWeight;
import org.grouplens.lenskit.core.Shareable;

@Shareable
public class LogWeight implements TDAWeight {
	@Override
	public double getWeight(int num) {
		double weight = 0;
		for(int i = 2; i <= num; i++){
			weight += Math.log(i) / Math.log(2.0);
		}
		return weight;
	}
}
