package online;

public class RatedMovie {
	private String id;
	private String name;
	private Boolean relevant;
	private Boolean novel;
	private Boolean unexpected;
	private double unpopularity;
	private double dissimilarity;
	private double likedDissimilarity;

	public RatedMovie(String id, String name, Boolean relevant, Boolean novel, Boolean unexpected) {
		this.id = id;
		this.name = name;
		this.relevant = relevant;
		this.novel = novel;
		this.unexpected = unexpected;
	}

	public String getId() {
		return id;
	}

	public double getLikedDissimilarity() {
		return likedDissimilarity;
	}

	public Integer getRelevant() {
		return relevant ? 1 : 0;
	}

	public Integer getNovel() {
		return novel ? 1 : 0;
	}

	public Integer getUnexpected() {
		return unexpected ? 1 : 0;
	}

	public String getName() {
		return name;
	}

	public double getUnpopularity() {
		return unpopularity;
	}

	public double getDissimilarity() {
		return dissimilarity;
	}

	public void setUnpopularity(double unpopularity) {
		this.unpopularity = unpopularity;
	}

	public void setDissimilarity(double dissimilarity) {
		this.dissimilarity = dissimilarity;
	}

	public void setLikedDissimilarity(double likedDissimilarity) {
		this.likedDissimilarity = likedDissimilarity;
	}
}
