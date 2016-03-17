public class Test {
	private static double learnParam = 0.0001;
	private static double lambda = 0.01;

	public static void main(String[] args) {
		maximize();
	}

	private static void maximize() {
		double x = 0.1, dx;
		double y = 0.1, dy;
		for (int i = 0; i < 10000; i++) {
			double f = 30 * x + y - lambda * Math.max(0, x + y - 1) - lambda * Math.max(0, -y);
			System.out.println("x " + x + " y " + y);
			System.out.println("Func " + f);
			f = 30 * x + y;
			System.out.println("True func " + f);
			if (x + y < 1) {
				dx = 30 * learnParam;
				dy = learnParam;
			} else {
				dx = 30 * learnParam - lambda;
				dy = learnParam - lambda;
			}
			if (y < 0) {
				dy += lambda;
			}
			x += dx;
			y += dy;
		}
	}
}
