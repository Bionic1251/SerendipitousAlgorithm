package diversity;

public class ExpWeight implements TDAWeight {
	@Override
	public double getWeight(int num) {
		double weight = 0;
		for (int i = 1; i <= num; i++) {
			weight += 1 / i;
		}
		return weight;
	}
}
