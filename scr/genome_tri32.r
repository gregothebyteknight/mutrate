


# Generate 32-trinucleotide context sequences for a given genome.
# In this case, we are using the human genome (hg19) as an example
# The output will be a csv file containing the probabilities of
# 32 trinucleotide with a pyrimidine in the middle position

library(Biostrings)
library(BSgenome.Hsapiens.UCSC.hg19)

chrs <- paste0("chr", c(1:22, "X", "Y"))
raw <- Reduce(`+`, lapply(
  chrs,
  function(ch) {
    Biostrings::trinucleotideFrequency(
      Hsapiens[[ch]]
    )
  }
)
)

raw_prob <- raw / sum(raw)

printf <- function(...) cat(sprintf(...))
printf("Sum of probabilities: %.6f\n", sum(raw_prob))

collapse32 <- function(raw) {
  tri <- names(raw)
  mid <- substr(tri, 2, 2)

  rc <- as.character(reverseComplement(DNAStringSet(tri)))

  label <- ifelse(mid %in% c("C", "T"), tri, rc)

  tapply(raw, label, sum)
}

tri32 <- collapse32(raw_prob)

write.csv(tri32, file = "../data/h19_tri32.csv")