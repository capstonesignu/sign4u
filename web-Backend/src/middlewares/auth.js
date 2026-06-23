const jwt= require('jsonwebtoken');

module.exports = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1]; // Bearer: {token}
    if (!token){
        return res.status(401).json({error: '인증이 필요합니다'});
        }
        
    try{
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.user = decoded; /// 이후 라우터에서 req.user로 유저 id 접근
        next();
    } catch (err){
        return res.status(401).json({error: "유효하지 않은 토큰입니다"});

    }
};
