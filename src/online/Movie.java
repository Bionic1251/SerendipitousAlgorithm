package online;

public class Movie {
	private String imdbId;
	private String id;
	private String name;
	private String year;
	private String genres;

	public Movie(String imdbId, String name) {
		this.imdbId = imdbId;
		this.name = name;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getId() {
		return id;
	}

	public void setYear(String year) {
		this.year = year;
	}

	public void setGenres(String genres) {
		this.genres = genres;
	}

	public String getImdbId() {
		return imdbId;
	}

	public String getName() {
		return name;
	}

	public String getYear() {
		return year;
	}

	public String getGenres() {
		return genres;
	}

	@Override
	public String toString() {
		return "Movie{" +
				"imdbId='" + imdbId + '\'' +
				", name='" + name + '\'' +
				", year='" + year + '\'' +
				", genres='" + genres + '\'' +
				'}';
	}
}
