package annotation;

import org.grouplens.grapht.annotation.DefaultNull;

import javax.inject.Qualifier;
import java.lang.annotation.*;

@Documented
@Qualifier
@DefaultNull
//@Parameter(ItemScorer.class)
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface RatingPredictor2 {
}
