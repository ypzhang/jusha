void exclusive_scan(const int *in_begin, const int *in_end, int *out_begin)
{
  if (begin == end)   return;
  int last_in = *in_begin++;
  *out_begin++ = 0;
  while (in_begin < in_end) {
    int new_v = *out + last_in;
    last_in = *in_begin++;
    *out_begin++ = new_v;
  }
}
