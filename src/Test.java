import lc.investigation.Maximizer;

public class Test {
	private static double learnParam = 0.0001;
	private static double lambda = 0.01;

	public static void main(String[] args) {
		maximize();
	}

	private static void maximize() {
		double serR = 5;//3046718.1666646022;
		double serD = 5;//2528437.980532453;
		double serU = 5;//3302601.673728431;

		double unserR = 4;//2276738.3333337605;
		double unserD = 1.9;//1438245.4855904642;
		double unserU = 1;//2778419.0418167175;

		Maximizer maximizer = new Maximizer(serR, serD, serU, unserR, unserD, unserU);
		maximizer.optimize();
	}
}
