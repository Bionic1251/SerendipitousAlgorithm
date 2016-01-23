package annotation;

import funkSVD.lu.LuFunkSVDUpdateRule;
import org.grouplens.grapht.annotation.DefaultNull;
import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;

@Documented
@Qualifier
@DefaultNull
@Parameter(LuFunkSVDUpdateRule.class)
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface UpdateRule {
}
