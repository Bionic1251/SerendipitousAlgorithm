package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.PrintWriter;
import java.util.*;

public class PrepareUtil {
	public static void printStatistics(String path) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				int i = 0;
				String line = reader.readLine();
				Set<Long> userSet = new HashSet<Long>();
				Set<Long> itemSet = new HashSet<Long>();
				while (line != null) {
					i++;
					String[] vec = line.split("\t");
					Long userId = Long.valueOf(vec[0]);
					userSet.add(userId);
					Long itemId = Long.valueOf(vec[1]);
					itemSet.add(itemId);
					line = reader.readLine();
				}
				System.out.println("Users " + userSet.size() + " Items " + itemSet.size() + " Ratings " + i);
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareYahooDataset(String path) {
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("ratings.dat"));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					String row = vector[0] + "\t" + vector[1] + "\t" + vector[3];
					writer.println(row);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareSmallDataset(String path) {
		int boolLen = 23, boolStart = 5;
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("small_content.dat"));
			try {
				String text;
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\\|");
					text = vector[0];
					for (int i = boolStart; i <= boolLen; i++) {
						text += "," + vector[i];
					}
					writer.println(text);
					line = reader.readLine();
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void prepareBigDataset(String path) {
		Map<String, Integer> attributeMap = getAttributesMap(path);
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			PrintWriter writer = new PrintWriter(new File("big_content.dat"));
			try {
				String line = reader.readLine();
				String[] vector = line.split("\t");
				String id = vector[0];
				int[] values = new int[attributeMap.size()];
				while (line != null) {
					vector = line.split("\t");
					String newId = vector[0];
					if (!newId.equals(id)) {
						saveItem(id, values, writer);
						id = newId;
						values = new int[attributeMap.size()];
						continue;
					} else {
						int index = attributeMap.get(vector[1]);
						values[index] = 1;
					}

					line = reader.readLine();
					if (line == null) {
						saveItem(id, values, writer);
					}
				}
			} finally {
				reader.close();
				writer.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void saveItem(String id, int[] values, PrintWriter writer) {
		String text = id;
		for (int i = 0; i < values.length; i++) {
			text += "," + values[i];
		}
		writer.println(text);
	}

	private static Map<String, Integer> getYahooAttributesMap(String path) {
		Set<String> set = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					String genres = vector[10];
					String[] genresVec = genres.split("\\|");
					for (String g : genresVec) {
						set.add(g);
					}
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		set.remove("\\N");
		set.remove("~Delete");
		List<String> list = new ArrayList<String>(set);
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}
		return map;
	}

	private static Map<String, Integer> getAttributesMap(String path) {
		Set<String> set = new HashSet<String>();
		try {
			BufferedReader reader = new BufferedReader(new java.io.FileReader(path));
			try {
				String line = reader.readLine();
				while (line != null) {
					String[] vector = line.split("\t");
					set.add(vector[1]);
					line = reader.readLine();
				}
			} finally {
				reader.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		List<String> list = new ArrayList<String>(set);
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < list.size(); i++) {
			map.put(list.get(i), i);
		}
		return map;
	}
}
