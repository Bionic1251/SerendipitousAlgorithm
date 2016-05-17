import org.grouplens.lenskit.ItemRecommender;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.dao.SimpleFileRatingDAO;
import org.grouplens.lenskit.scored.ScoredId;
import util.AlgorithmUtil;
import util.ContentAverageDissimilarity;
import util.Settings;
import util.Util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class RecommenderRunner {
	public static void main(String args[]) throws Exception {
		Util.setParameters();
		ContentAverageDissimilarity.create(Settings.DATASET_CONTENT);
		PrintWriter writer = new PrintWriter(new File("res.txt"));

		List<Long> userList = new ArrayList<Long>();
		/*userList.add(75l);
		userList.add(78l);
		userList.add(170l);*/
		userList.add(1l);
		userList.add(2l);
		userList.add(3l);
		/*for (long i = 100029l; i < 100033l; i++) {
			userList.add(i);
		}
		userList.add(100040l);*/
		getRecommendations(userList, writer);

		writer.close();
	}

	private static void getRecommendations(List<Long> userList, PrintWriter writer) throws Exception {
		printRecs(AlgorithmUtil.getTDASVD(0.9), userList, writer, "TDASVD");
		printRecs(AlgorithmUtil.getSVD(), userList, writer, "SVD");
		//printRecs(AlgorithmUtil.getFilterSerSVD(), userList, writer, "FilterSerSVD");
		//printRecs(AlgorithmUtil.getFilterSVD(), userList, writer, "FilterSVD");
		//printRecs(AlgorithmUtil.getFunkSVD(), userList, writer, "FunkSVD");
		//printRecs(AlgorithmUtil.getSerPop(), userList, writer, "SerPop");
		//printRecs(AlgorithmUtil.getSerUB(), userList, writer, "SerUB");
		//printRecs(AlgorithmUtil.getSerTFIDF(), userList, writer, "SerTFIDF");
		//printRecs(AlgorithmUtil.getSerContent(), userList, writer, "SerContent");
		//printRecs(AlgorithmUtil.getTFIDF(), userList, writer, "tfidf");
		//printRecs(AlgorithmUtil.getContent(), userList, writer, "Content");
		//printRecs(AlgorithmUtil.getCompletelyRandom(), userList, writer, "Random");
		//printRecs(AlgorithmUtil.getSVD(), userList, writer, "SVD");
		//printRecs(AlgorithmUtil.getLCRDU(), userList, writer, "LCRDU");
		//printRecs(AlgorithmUtil.getLCRD(), userList, writer, "LCRD");
		//printRecs(AlgorithmUtil.getLCRU(), userList, writer, "LCRU");
		//printRecs(AlgorithmUtil.getLuSVDHinge10000(), userList, writer, "SPR");
		//printRecs(AlgorithmUtil.getZhengSVD(), userList, writer, "Zheng");

		/*System.out.println("PureSVD");
		writer.println("PureSVD");
		printRecs(AlgorithmUtil.getPureSVD(), userList, writer);*/

		/*System.out.println("Random");
		writer.println("Random");
		printRecs(AlgorithmUtil.getRandom(), userList, writer);*/

		/*System.out.println("SPR");
		writer.println("SPR");
		printRecs(AlgorithmUtil.getLuSVDHinge10000(), userList, writer);*/
	}

	private static void printRecs(LenskitConfiguration configuration, List<Long> userList, PrintWriter writer, String algName) throws Exception {
		configuration.bind(EventDAO.class).to(new SimpleFileRatingDAO(new File(Settings.DATASET), "\t"));
		LenskitRecommender pop = LenskitRecommender.build(configuration);
		ItemRecommender itemRecommender = pop.getItemRecommender();
		String out = "";
		for (Long userId : userList) {
			out += algName + "\t" + userId;
			List<ScoredId> recs = itemRecommender.recommend(userId, 3000);
			out += "\t";
			for (ScoredId scoredId : recs) {
				out += scoredId.getId() + "=" + scoredId.getScore() + ",";
			}
			out = out.substring(0, out.length() - 1);
			out += "\r\n";
		}
		System.out.print(out);
		writer.print(out);
	}

}
