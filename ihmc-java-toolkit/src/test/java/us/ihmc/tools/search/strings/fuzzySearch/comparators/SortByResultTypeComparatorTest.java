package us.ihmc.tools.search.strings.fuzzySearch.comparators;

import static us.ihmc.robotics.Assert.*;

import java.util.Random;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Disabled;
import us.ihmc.tools.search.strings.fuzzySearch.CombinedFuzzySearchResult;
import us.ihmc.tools.search.strings.fuzzySearch.CombinedFuzzySearchResult.CombinedFuzzySearchResultType;

/**
 * @author Doug Stephen <a href="mailto:dstephen@ihmc.us">(dstephen@ihmc.us)</a>
 */
public class SortByResultTypeComparatorTest
{
   @Test
   public void testCompare()
   {
      Random random = new Random(1976L);
      SortByResultTypeComparator comparator = new SortByResultTypeComparator();

      CombinedFuzzySearchResult aardvark = new CombinedFuzzySearchResult("Aardvark", CombinedFuzzySearchResultType.EXACT_SUBSTRING);
      CombinedFuzzySearchResult burgundy = new CombinedFuzzySearchResult("Burgundy", CombinedFuzzySearchResultType.REGEX);
      CombinedFuzzySearchResult sebastopol = new CombinedFuzzySearchResult("Sebastopol", CombinedFuzzySearchResultType.FUZZY);

      for (int i = 0; i < 10000; i++)
      {
         CombinedFuzzySearchResultType randomResultType = generateRandomResultType(random);
         CombinedFuzzySearchResult victorious = new CombinedFuzzySearchResult("Victorious", randomResultType);

         int exactSubstringCompare = comparator.compare(aardvark, victorious);
         int regexCompare = comparator.compare(burgundy, victorious);
         int fuzzyCompare = comparator.compare(sebastopol, victorious);

         switch (randomResultType)
         {
         case FUZZY:
            assertTrue(exactSubstringCompare < 0);
            assertTrue(regexCompare < 0);
            assertTrue(fuzzyCompare == 0);
            break;
         case REGEX:
            assertTrue(exactSubstringCompare < 0);
            assertTrue(regexCompare == 0);
            assertTrue(fuzzyCompare > 0);
            break;
         case EXACT_SUBSTRING:
            assertTrue(exactSubstringCompare == 0);
            assertTrue(regexCompare > 0);
            assertTrue(fuzzyCompare > 0);
            break;
         }
      }

   }

   private static CombinedFuzzySearchResultType generateRandomResultType(Random random)
   {
      int numberOfEnums = CombinedFuzzySearchResultType.class.getEnumConstants().length;
      return CombinedFuzzySearchResultType.class.getEnumConstants()[random.nextInt(numberOfEnums)];
   }
}