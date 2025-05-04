import  { Box, Typography } from "@mui/material";

const Results = () => {
  return (
    <Box sx={{ padding: 2, height: "100%" }}>
      <Box
        sx={{
          height: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Typography variant="h5">Waiting for results...</Typography>
      </Box>
      <Box
        sx={{
          height: "100%",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Typography variant="h5">Results will be displayed here</Typography>
      </Box>
    </Box>
  );
};

export default Results;
