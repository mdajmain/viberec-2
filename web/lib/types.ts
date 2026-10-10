export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  premise: string;
  poster: string | null;
  why?: string;
  why_source?: string;
};

export type QuizState = {
  done: boolean;
  likes: number;
  answered: number;
  target_likes: number;
  cap: number;
  movie: Movie | null;
};

export type Feed = { feed_id: string; personal: boolean; results: Movie[] };

export type Understood = {
  sample_movies: string[];
  genres: string[];
  more_of: string[];
  less_of: string[];
  about: string[];
};

export type SearchResult = { search_id: string; prompt: string; understood: Understood; results: Movie[] };

export type Taste = {
  likes: number;
  dislikes: number;
  unseen: number;
  top_vibes: string[];
  light_heavy: string | null;
  pacing: string | null;
  era: string | null;
  top_genres: string[];
};

export type ShownFilm = { movie_id: number; position: number; title: string; year: number | null; poster: string | null };

export type History = {
  searches: { search_id: string; prompt: string; understood: Understood; created: string; results: ShownFilm[] }[];
  reactions: {
    movie_id: number;
    title: string;
    year: number | null;
    poster: string | null;
    verdict: Verdict;
    source: "quiz" | "search" | "feed";
    created: string;
  }[];
};

export type Verdict = "like" | "dislike" | "unseen" | "not_this_vibe";
