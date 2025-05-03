const { Box } = require("@mui/material")

const Results = () =>{
    return (
        <Box sx={{padding: 2,  height:'100%'}}>
            <Box sx={{height:'100%', display:'flex', justifyContent:'center', alignItems:'center'}}>
                <h1>Waiting for results...</h1>
            </Box>
            <Box sx={{height:'100%', display:'flex', justifyContent:'center', alignItems:'center'}}>
                <h1>Results will be displayed here</h1>
            </Box>

        </Box>
    )
}

export default Results;